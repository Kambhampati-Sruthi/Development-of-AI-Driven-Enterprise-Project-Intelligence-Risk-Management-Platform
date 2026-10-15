from pathlib import Path
import hashlib

import chromadb
import pandas as pd
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent
        self.upload_dir = self.root / "data" / "uploads"
        self.db_dir = self.root / "data" / "chroma_db"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name="project_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def _extract_text(self, path):
        path = Path(path)
        extension = path.suffix.lower()
        sections = []

        if extension == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    sections.append(
                        {"text": text, "page": page_number}
                    )

        elif extension == ".docx":
            document = Document(str(path))
            text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            )
            if text.strip():
                sections.append({"text": text})

        elif extension == ".csv":
            dataframe = pd.read_csv(path).fillna("")
            for row_number, row in enumerate(
                dataframe.to_dict(orient="records"), start=1
            ):
                row_text = " | ".join(
                    f"{key}: {value}" for key, value in row.items()
                )
                if row_text.strip():
                    sections.append(
                        {"text": row_text, "row": row_number}
                    )

        elif extension in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8-sig")
            if text.strip():
                sections.append({"text": text})

        else:
            raise ValueError(
                "Unsupported file type. Use PDF, DOCX, CSV, TXT, or MD."
            )

        if not sections:
            raise ValueError(
                "No readable text was found. If this is a scanned PDF, "
                "you may need OCR before indexing it."
            )

        return sections

    def _make_chunks(self, sections, chunk_size=1200, overlap=180):
        chunks = []

        for section in sections:
            text = section["text"]
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()

                if chunk:
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            key: value
                            for key, value in section.items()
                            if key != "text"
                        },
                    })

                if end >= len(text):
                    break

                start = end - overlap

        return chunks

    def index_file(self, file_path):
        path = Path(file_path)
        source = path.name

        sections = self._extract_text(path)
        chunks = self._make_chunks(sections)

        if not chunks:
            raise ValueError("No text chunks were created.")

        # Remove older chunks for this filename before re-indexing.
        existing = self.collection.get(
            where={"source": source},
            include=[],
        )
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        ids = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            raw_id = f"{source}:{index}:{chunk['text']}"
            chunk_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()

            metadata = {
                "source": source,
                "chunk_number": index + 1,
                "file_type": path.suffix.lower().lstrip("."),
            }
            metadata.update(chunk["metadata"])

            ids.append(chunk_id)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        return {
            "source": source,
            "chunks": len(chunks),
            "status": "Indexed successfully",
        }

    def search(self, query, n_results=5):
        query = query.strip()

        if not query:
            return []

        total = self.collection.count()
        if total == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for text, metadata, distance in zip(
            documents, metadatas, distances
        ):
            matches.append({
                "text": text,
                "metadata": metadata or {},
                "distance": distance,
            })

        return matches

    def get_dashboard_stats(self):
        result = self.collection.get(
            include=["documents", "metadatas"]
        )
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []

        sources = {
            item.get("source", "Unknown")
            for item in metadatas
        }

        risk_terms = (
            "blocked", "blocker", "delay", "delayed", "overdue",
            "risk", "failed", "failure", "dependency", "延期",
            "critical", "pending", "slippage", "issue",
        )

        risks = []
        for text, metadata in zip(documents, metadatas):
            lowered = text.lower()
            matched = [term for term in risk_terms if term in lowered]

            if matched:
                risks.append({
                    "source": metadata.get("source", "Unknown"),
                    "text": text,
                    "signals": matched,
                })

        chunk_count = len(documents)
        risk_count = len(risks)

        # Preliminary keyword-based indicator, not an AI risk prediction.
        health_score = (
            100 if chunk_count == 0
            else max(0, min(100, 100 - round(
                60 * min(risk_count / max(chunk_count, 1), 1)
            )))
        )

        return {
            "document_count": len(sources),
            "chunk_count": chunk_count,
            "risk_count": risk_count,
            "health_score": health_score,
            "risks": risks[:20],
        }