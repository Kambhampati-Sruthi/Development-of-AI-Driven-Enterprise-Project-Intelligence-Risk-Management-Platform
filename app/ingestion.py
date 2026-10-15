
from io import BytesIO
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pandas as pd
from docx import Document
from pypdf import PdfReader
from pptx import Presentation


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".log", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c",
    ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".php",
    ".rb", ".swift", ".kt", ".sql", ".sh", ".bat", ".ps1",
    ".css", ".scss", ".ini", ".cfg", ".conf", ".toml",
    ".r", ".ipynb.txt",
}


class _HTMLTextParser(HTMLParser):
    """Extract readable text from HTML markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.parts.append(text)


def _record(text, source, **metadata):
    """Create a normalized text record with source metadata."""
    text = str(text).strip()

    if not text:
        return None

    return {
        "text": text,
        "source": source,
        **metadata,
    }


def _decode_text(content):
    """Decode text bytes with common encoding fallbacks."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="replace")


def _extract_tabular_records(frame, source, sheet=None):
    """Convert dataframe rows into searchable text records."""
    records = []
    frame = frame.fillna("")

    for row_number, row in enumerate(
        frame.to_dict(orient="records"), start=2
    ):
        fields = [
            f"{key}: {value}"
            for key, value in row.items()
            if str(value).strip()
        ]

        text = " | ".join(fields)

        if sheet is not None:
            text = f"Sheet: {sheet}\n{text}"

        metadata = {"row": row_number}
        if sheet is not None:
            metadata["sheet"] = str(sheet)

        item = _record(text, source, **metadata)
        if item:
            records.append(item)

    return records


def extract_document(filename, content):
    """
    Extract readable text records from a supported file.

    Args:
        filename: Original filename, including its extension.
        content: File content as bytes.

    Returns:
        A list of dictionaries containing text, source, and metadata.

    Supported formats:
        PDF, DOCX, CSV, TSV, XLSX, XLS, JSON, HTML, XML,
        PPTX, and common text/source-code formats.
    """
    filename = Path(str(filename)).name
    suffix = Path(filename).suffix.lower()
    source = filename
    records = []

    if not content:
        raise ValueError("The uploaded file is empty.")

    try:
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError(
                        "The PDF is password-protected."
                    ) from exc

            for page_number, page in enumerate(reader.pages, start=1):
                item = _record(
                    page.extract_text() or "",
                    source,
                    page=page_number,
                )
                if item:
                    records.append(item)

        elif suffix == ".docx":
            doc = Document(BytesIO(content))

            for index, paragraph in enumerate(
                doc.paragraphs, start=1
            ):
                item = _record(
                    paragraph.text,
                    source,
                    paragraph=index,
                )
                if item:
                    records.append(item)

            for table_index, table in enumerate(
                doc.tables, start=1
            ):
                for row_index, row in enumerate(
                    table.rows, start=1
                ):
                    text = " | ".join(
                        cell.text.strip() for cell in row.cells
                    )

                    item = _record(
                        text,
                        source,
                        table=table_index,
                        row=row_index,
                    )
                    if item:
                        records.append(item)

        elif suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","

            frame = pd.read_csv(
                BytesIO(content),
                sep=separator,
                encoding="utf-8-sig",
                encoding_errors="replace",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="error",
            )

            records.extend(
                _extract_tabular_records(frame, source)
            )

        elif suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(
                BytesIO(content),
                sheet_name=None,
                dtype=str,
            )

            for sheet_name, frame in sheets.items():
                records.extend(
                    _extract_tabular_records(
                        frame,
                        source,
                        sheet=sheet_name,
                    )
                )

        elif suffix == ".json":
            parsed = json.loads(_decode_text(content))

            if isinstance(parsed, list):
                for index, entry in enumerate(parsed, start=1):
                    item = _record(
                        json.dumps(
                            entry,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        source,
                        record=index,
                    )
                    if item:
                        records.append(item)
            else:
                item = _record(
                    json.dumps(
                        parsed,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    source,
                )
                if item:
                    records.append(item)

        elif suffix in {".html", ".htm"}:
            parser = _HTMLTextParser()
            parser.feed(_decode_text(content))
            parser.close()

            item = _record(
                "\n".join(parser.parts),
                source,
            )
            if item:
                records.append(item)

        elif suffix in {".xml", ".xsd", ".xsl"}:
            root = ET.fromstring(content)
            parts = []

            for element in root.iter():
                tag = element.tag.split("}")[-1]

                if element.text and element.text.strip():
                    parts.append(
                        f"{tag}: {element.text.strip()}"
                    )

                for key, value in element.attrib.items():
                    parts.append(f"{key}: {value}")

            item = _record("\n".join(parts), source)
            if item:
                records.append(item)

        elif suffix == ".pptx":
            presentation = Presentation(BytesIO(content))

            for slide_number, slide in enumerate(
                presentation.slides, start=1
            ):
                parts = []

                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        if shape.text.strip():
                            parts.append(shape.text.strip())

                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            parts.append(
                                " | ".join(
                                    cell.text.strip()
                                    for cell in row.cells
                                )
                            )

                item = _record(
                    "\n".join(parts),
                    source,
                    slide=slide_number,
                )
                if item:
                    records.append(item)

        elif suffix in TEXT_EXTENSIONS:
            item = _record(_decode_text(content), source)
            if item:
                records.append(item)

        else:
            raise ValueError(
                f"Unsupported file type: "
                f"{suffix or '(no extension)'}. "
                "Supported formats include PDF, DOCX, CSV, TSV, "
                "TXT, MD, JSON, XLSX, XLS, PPTX, HTML, XML, YAML, "
                "logs, and common source-code files."
            )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read '{filename}': {exc}"
        ) from exc

    if not records:
        raise ValueError(
            "No readable text was extracted from this file. "
            "It may be empty, image-only, or require OCR."
        )

    return records


def extract_text_from_file(file_path):
    """
    Backward-compatible path-based extractor.

    Allows existing dashboard code to call:
        extract_text_from_file(file_path)

    Returns all extracted text as one string.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records = extract_document(
        filename=path.name,
        content=path.read_bytes(),
    )

    return "\n\n".join(
        record["text"]
        for record in records
        if record.get("text")
    )
