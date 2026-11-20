"""Document ingestion: robust multi-format parsing into a uniform document model.

Every parser returns a ``ParsedDocument`` made of text ``Segment`` objects (for
narrative content) and ``TableData`` objects (for spreadsheets/CSVs). The rest of
the platform never needs to know which file format the content came from.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .config import MAX_FILE_MB, MAX_TABLE_ROWS, SUPPORTED_EXTENSIONS
from .utils import clean, relabel_name, relabel_text, short_hash

log = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised with a user-friendly message when a file cannot be ingested."""


# ------------------------------------------------------------------ data model --
@dataclass
class Segment:
    text: str
    locator: str                  # e.g. "Page 3", "§ Risks", "Sheet 'UT', record 4"
    row_no: Optional[int] = None  # set for table rows so the chunker can group them


@dataclass
class TableData:
    sheet: str
    df: pd.DataFrame
    kind: str                     # backlog | defects | tests | generic
    roles: Dict[str, str]         # semantic role -> column name


@dataclass
class ParsedDocument:
    name: str
    ext: str
    size_bytes: int
    segments: List[Segment] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    doc_id: str = ""

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.segments)

    @property
    def is_tabular(self) -> bool:
        return bool(self.tables)

    @property
    def kind_label(self) -> str:
        if self.tables:
            kinds = {t.kind for t in self.tables}
            for k, lab in (("defects", "Defect tracker"), ("tests", "Test plan"), ("backlog", "Sprint backlog")):
                if k in kinds:
                    return lab
            return "Table"
        return {"pdf": "PDF document", "docx": "Word document", "txt": "Text notes", "md": "Text notes"}.get(
            self.ext, "Document")


# --------------------------------------------------------------- column roles ---
_ROLE_SYNONYMS = [
    ("expected", ["expected result", "expected outcome"]),
    ("actual", ["actual result", "actual outcome", "actual"]),
    ("action_date", ["action taken date"]),
    ("action", ["action taken", "resolution"]),
    ("owner", ["assigned to", "assignee", "owner", "responsible", "team member", "resource", "developer", "tester"]),
    ("dependency", ["dependency", "depends on", "dependent"]),
    ("impediment", ["impediment", "blocker", "issues faced", "issue"]),
    ("status", ["status", "state", "progress", "stage"]),
    ("sprint", ["sprint", "iteration", "release"]),
    ("priority", ["priority", "severity", "criticality"]),
    ("estimate", ["story point", "points", "estimate", "effort", "hours", "size"]),
    ("start", ["start"]),
    ("due", ["due", "end date", "deadline", "target date", "finish", "planned end"]),
    ("type", ["type", "category"]),
    ("remarks", ["remark", "comment", "note"]),
    ("title", ["user story description", "story description", "task description", "user story", "story", "summary", "task", "title", "backlog item", "feature", "requirement",
               "test case name", "test case", "activity", "deliverable", "description", "item"]),
    ("date", ["date"]),
]


def _norm(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower())


def map_columns(columns: List[str]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    used: set = set()
    for role, syns in _ROLE_SYNONYMS:
        for syn in syns:
            hit = next((c for c in columns if c not in used and syn in _norm(c)), None)
            if hit:
                roles[role] = hit
                used.add(hit)
                break
    return roles


def classify_table(columns: List[str], roles: Dict[str, str]) -> str:
    joined = " | ".join(_norm(c) for c in columns)
    if "expected" in roles and "actual" in roles:
        return "tests"
    if any(w in joined for w in ("defect", "bug", "detected")):
        return "defects"
    if "title" in roles and any(r in roles for r in ("status", "sprint", "owner", "estimate", "priority")):
        return "backlog"
    return "generic"


# ----------------------------------------------------------------- table build --
def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, pd.Timestamp)):
        if pd.isna(v):
            return ""
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if v.is_integer():
            return str(int(v))
    return clean(v)


def build_table(raw: pd.DataFrame, sheet: str) -> Optional[TableData]:
    """Turn a header-less raw sheet into a clean table (detects the header row)."""
    if raw is None or raw.empty:
        return None
    raw = raw.map(_cell) if hasattr(raw, "map") else raw.applymap(_cell)
    raw = raw.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all").fillna("")
    raw = raw.reset_index(drop=True)
    if raw.shape[1] < 2 or raw.empty:
        return None

    def text_cells(row) -> int:
        return sum(1 for v in row if v and not re.fullmatch(r"[-+]?[\d.,:/\- ]+", str(v)))

    scan = min(15, len(raw))
    scores = [text_cells(raw.iloc[i]) for i in range(scan)]
    best = max(scores)
    if best < 2:
        return None
    header_idx = next(i for i, s in enumerate(scores) if s >= max(2, int(best * 0.8)))

    cols: List[str] = []
    for i, v in enumerate(raw.iloc[header_idx]):
        name = clean(v) or f"Column {i + 1}"
        while name in cols:
            name += "_"
        cols.append(name)
    body = raw.iloc[header_idx + 1:].copy()
    body.columns = cols
    body = body.replace("", pd.NA).dropna(how="all").fillna("")
    min_cells = 2 if body.shape[1] >= 3 else 1             # one-cell rows are section labels, not records
    keep = body.apply(lambda r: any(v and not re.fullmatch(r"[-+]?[\d.,:/\- ]+", str(v)) for v in r)
                      and sum(1 for v in r if v) >= min_cells, axis=1)
    body = body[keep].head(MAX_TABLE_ROWS).reset_index(drop=True)
    if body.empty:
        return None
    roles = map_columns(cols)
    return TableData(sheet=sheet, df=body, kind=classify_table(cols, roles), roles=roles)


_NOISE_COL = re.compile(r"^day\s*\d+$", re.I)      # burn-down grid columns carry no narrative value


def row_to_text(row: pd.Series) -> str:
    return "; ".join(f"{c}: {v}" for c, v in row.items() if clean(v) and not _NOISE_COL.match(str(c)))


def _tables_to_segments(tables: List[TableData]) -> List[Segment]:
    segs: List[Segment] = []
    for t in tables:
        segs.append(Segment(
            f"Table '{t.sheet}' ({t.kind}) with columns: {', '.join(t.df.columns)}. {len(t.df)} records.",
            f"Sheet '{t.sheet}', header"))
        for i, (_, row) in enumerate(t.df.iterrows(), start=1):
            txt = row_to_text(row)
            if txt:
                segs.append(Segment(f"[{t.sheet}] {txt}", f"Sheet '{t.sheet}', record {i}", row_no=i))
    return segs


# --------------------------------------------------------------------- parsers --
def _parse_pdf(data: bytes, doc: ParsedDocument) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("PDF support needs the 'pypdf' package (pip install pypdf).") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise IngestionError("This PDF is password-protected. Remove the password and upload it again.")
        empty_pages = 0
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                doc.warnings.append(f"Page {i} could not be read ({exc.__class__.__name__}).")
                continue
            text = "\n".join(clean(l) for l in text.splitlines() if clean(l))
            if text:
                doc.segments.append(Segment(text, f"Page {i}"))
            else:
                empty_pages += 1
        if empty_pages and doc.segments:
            doc.warnings.append(f"{empty_pages} page(s) had no extractable text (images or scans).")
        if not doc.segments:
            raise IngestionError("No text layer found in this PDF — it looks like a scan. Run OCR first.")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"The PDF could not be parsed: {exc}") from exc


def _parse_docx(data: bytes, doc: ParsedDocument) -> None:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("DOCX support needs 'python-docx' (pip install python-docx).") from exc
    try:
        d = Document(io.BytesIO(data))
    except Exception as exc:
        raise IngestionError(f"The Word document is corrupted or unsupported: {exc}") from exc

    heading = "Opening"
    lines: List[str] = []

    def flush() -> None:
        body = "\n".join(l for l in lines if l.strip())
        if body.strip():
            doc.segments.append(Segment(body, f"§ {heading}"))
        lines.clear()

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = clean(p.text)
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            runs = [r for r in p.runs if clean(r.text)]
            all_bold = bool(runs) and all(bool(r.bold) for r in runs)
            is_heading = style.lower().startswith(("heading", "title")) or (all_bold and len(text) <= 90)
            numbered = child.pPr is not None and child.pPr.numPr is not None
            if is_heading:
                flush()
                heading = text.rstrip(":").strip("* ")
                lines.append(f"## {heading}")
            elif style.lower().startswith("list") or numbered or text[:1] in "•-–*▪":
                lines.append("• " + text.lstrip("•-–*▪ ").strip())
            else:
                lines.append(text)
        elif tag == "tbl":
            try:
                table = Table(child, d)
                for r in table.rows:
                    cells = []
                    for c in r.cells:
                        t = clean(c.text)
                        if t and (not cells or cells[-1] != t):   # merged cells repeat text
                            cells.append(t)
                    if cells:
                        lines.append(" | ".join(cells))
            except Exception as exc:
                doc.warnings.append(f"A table could not be read ({exc.__class__.__name__}).")
    flush()
    if not doc.segments:
        raise IngestionError("The Word document contains no readable text.")


def _parse_txt(data: bytes, doc: ParsedDocument) -> None:
    text = _decode(data)
    if not text.strip():
        raise IngestionError("The text file is empty.")
    blocks = re.split(r"\n\s*\n", text)
    for i, b in enumerate(blocks, start=1):
        b = "\n".join(clean(l) for l in b.splitlines() if clean(l))
        if b:
            doc.segments.append(Segment(b, f"Block {i}"))


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _parse_csv(data: bytes, doc: ParsedDocument) -> None:
    text = _decode(data)
    if not text.strip():
        raise IngestionError("The CSV file is empty.")
    raw = None
    for kwargs in ({"sep": None, "engine": "python"}, {"sep": ","}, {"sep": "\t"}, {"sep": ";"}):
        try:
            raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False,
                              on_bad_lines="skip", **kwargs)
            if raw.shape[1] >= 2:
                break
        except Exception:
            raw = None
    if raw is None or raw.empty:
        raise IngestionError("The CSV could not be parsed (check the delimiter and quoting).")
    table = build_table(raw, os.path.splitext(doc.name)[0])
    if table is None:
        raise IngestionError("No tabular data with a header row was found in this CSV.")
    doc.tables.append(table)
    doc.segments.extend(_tables_to_segments(doc.tables))


def _libreoffice_convert(data: bytes, ext: str) -> Optional[bytes]:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, f"input.{ext}")
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            subprocess.run([exe, "--headless", "--convert-to", "xlsx", "--outdir", tmp, src],
                           check=True, timeout=120, capture_output=True)
            out = os.path.join(tmp, "input.xlsx")
            if os.path.exists(out):
                with open(out, "rb") as fh:
                    return fh.read()
        except Exception as exc:
            log.warning("LibreOffice conversion failed: %s", exc)
    return None


def _read_excel_frames(data: bytes, ext: str) -> Dict[str, pd.DataFrame]:
    head = data[:8]
    is_zip = head.startswith(b"PK")
    is_ole = head.startswith(b"\xd0\xcf\x11\xe0")
    errors: List[str] = []

    def attempt(engine: str, payload: bytes):
        try:
            return pd.read_excel(io.BytesIO(payload), sheet_name=None, header=None, dtype=object, engine=engine)
        except ImportError as exc:
            errors.append(f"{engine}: {exc}")
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
        return None

    frames = None
    if is_zip:                                   # real xlsx (even if named .xls)
        frames = attempt("openpyxl", data)
    elif is_ole or ext == "xls":
        frames = attempt("xlrd", data)
        if frames is None:
            converted = _libreoffice_convert(data, "xls")
            if converted:
                frames = attempt("openpyxl", converted)
    if frames is None and not is_zip and not is_ole:   # HTML / delimited text saved as .xls
        text = _decode(data)
        try:
            tables = pd.read_html(io.StringIO(text), header=None)
            if tables:
                frames = {f"Sheet{i + 1}": t.astype(object) for i, t in enumerate(tables)}
        except Exception:
            try:
                frames = {"Sheet1": pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None,
                                                dtype=object)}
            except Exception as exc:
                errors.append(f"text fallback: {exc}")
    if frames is None:
        hint = ""
        if any("xlrd" in e for e in errors):
            hint = " Install the legacy-Excel reader with `pip install xlrd`, or save the file as .xlsx."
        raise IngestionError("The spreadsheet could not be read." + hint)
    return frames


def _parse_excel(data: bytes, doc: ParsedDocument) -> None:
    frames = _read_excel_frames(data, doc.ext)
    for sheet, raw in frames.items():
        try:
            table = build_table(raw, str(sheet))
        except Exception as exc:
            doc.warnings.append(f"Sheet '{sheet}' skipped ({exc.__class__.__name__}).")
            continue
        if table is None:
            doc.warnings.append(f"Sheet '{sheet}' skipped (no tabular data — likely a lookup list or blank).")
            continue
        doc.tables.append(table)
    if not doc.tables:
        raise IngestionError("No usable tables were found in this workbook.")
    doc.segments.extend(_tables_to_segments(doc.tables))


_PARSERS = {"pdf": _parse_pdf, "docx": _parse_docx, "txt": _parse_txt, "md": _parse_txt,
            "csv": _parse_csv, "xlsx": _parse_excel, "xls": _parse_excel}


def parse_file(name: str, data: bytes) -> ParsedDocument:
    """Parse one uploaded file. Raises ``IngestionError`` with a readable message."""
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '.{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}.")
    if not data:
        raise IngestionError("The file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise IngestionError(f"The file exceeds the {MAX_FILE_MB} MB limit.")
    name = relabel_name(name)
    doc = ParsedDocument(name=name, ext=ext, size_bytes=len(data), doc_id=short_hash(name, str(len(data))))
    try:
        _PARSERS[ext](data, doc)
    except IngestionError:
        raise
    except Exception as exc:  # last line of defence — never crash the UI
        log.exception("Unexpected parse failure for %s", name)
        raise IngestionError(f"Unexpected error while reading the file: {exc}") from exc
    for t in doc.tables:                                    # normalise phase wording inside tracker cells too
        t.df = t.df.apply(lambda col: col.map(lambda v: relabel_text(v) if isinstance(v, str) else v))
    for seg in doc.segments:
        seg.text = relabel_text(seg.text)
        seg.locator = relabel_text(seg.locator)
    return doc
