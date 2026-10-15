
from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion import extract_document


class RAGPipeline:
    """Local RAG pipeline for project document intelligence."""

    def __init__(self, upload_dir=None, chroma_dir=None):
        self.root = Path(__file__).resolve().parent.parent

        self.upload_dir = Path(
            upload_dir or self.root / "data" / "uploads"
        )
        self.db_dir = Path(
            chroma_dir or self.root / "data" / "chroma_db"
        )

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Local embedding model; no API key is required.
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
        """Use the shared ingestion module for document extraction."""
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        records = extract_document(
            filename=path.name,
            content=path.read_bytes(),
        )

        sections = []

        for record in records:
            text = record.get("text", "").strip()

            if not text:
                continue

            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"text", "source"}
                and isinstance(value, (str, int, float, bool))
            }

            sections.append({
                "text": text,
                **metadata,
            })

        if not sections:
            raise ValueError(
                f"No readable text found in '{path.name}'."
            )

        return sections

    def _make_chunks(
        self,
        sections,
        chunk_size=1200,
        overlap=180,
    ):
        """Split extracted text into overlapping character chunks."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller "
                "than chunk_size."
            )

        chunks = []

        for section in sections:
            text = section["text"]
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
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
        """Extract, chunk, embed, and index one document."""
        path = Path(file_path)
        source = path.name

        sections = self._extract_text(path)
        chunks = self._make_chunks(sections)

        if not chunks:
            raise ValueError(
                f"No text chunks were created for '{source}'."
            )

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        ids = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(
                chunk["text"].encode("utf-8")
            ).hexdigest()

            chunk_id = hashlib.sha256(
                f"{source}:{index}:{content_hash}".encode("utf-8")
            ).hexdigest()

            metadata = {
                "source": source,
                "chunk_number": index + 1,
                "file_type": path.suffix.lower().lstrip("."),
            }
            metadata.update(chunk["metadata"])

            ids.append(chunk_id)
            metadatas.append(metadata)

        # Replace older indexed chunks for this filename.
        # Index new content first so extraction/embedding failures
        # do not erase the previous index.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        # Remove stale chunks left over from a previous version
        # of the same file, if the chunk count has changed.
        current_ids = set(ids)
        existing = self.collection.get(
            where={"source": source},
            include=[],
        )
        stale_ids = [
            item_id
            for item_id in existing.get("ids", [])
            if item_id not in current_ids
        ]

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return {
            "source": source,
            "chunks": len(chunks),
            "status": "Indexed successfully",
        }

    def index_files(self, file_paths):
        """Index multiple files and report each result independently."""
        results = []

        for file_path in file_paths:
            try:
                results.append(self.index_file(file_path))
            except Exception as exc:
                results.append({
                    "source": Path(file_path).name,
                    "chunks": 0,
                    "status": f"Failed: {exc}",
                })

        return results

    def search(self, query, n_results=5):
        """Retrieve the most relevant indexed text chunks."""
        query = str(query).strip()

        if not query:
            return []

        if n_results < 1:
            return []

        total = self.collection.count()

        if total == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, total),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        for text, metadata, distance in zip(
            documents, metadatas, distances
        ):
            matches.append({
                "text": text,
                "metadata": metadata or {},
                "distance": distance,
            })

        return matches

    def retrieve_context(self, query, n_results=5):
        """Return retrieved context with source attribution for an assistant."""
        matches = self.search(query, n_results=n_results)

        if not matches:
            return {
                "query": query,
                "context": "",
                "sources": [],
                "matches": [],
            }

        context_parts = []
        sources = []

        for match in matches:
            metadata = match["metadata"]
            source = metadata.get("source", "Unknown document")

            location = ""
            if "page" in metadata:
                location = f", page {metadata['page']}"
            elif "sheet" in metadata:
                location = f", sheet {metadata['sheet']}"
                if "row" in metadata:
                    location += f", row {metadata['row']}"
            elif "row" in metadata:
                location = f", row {metadata['row']}"

            context_parts.append(
                f"Source: {source}{location}\n"
                f"{match['text']}"
            )

            source_info = {
                "source": source,
                "location": location.lstrip(", "),
            }

            if source_info not in sources:
                sources.append(source_info)

        return {
            "query": query,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "matches": matches,
        }

    def get_dashboard_stats(self):
        """Return knowledge-base metrics and preliminary keyword risk signals."""
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
            "blocked",
            "blocker",
            "delay",
            "delayed",
            "overdue",
            "risk",
            "failed",
            "failure",
            "dependency",
            "critical",
            "pending",
            "slippage",
            "issue",
            "escalation",
            "behind schedule",
            "延期",
        )

        risks = []

        for text, metadata in zip(documents, metadatas):
            lowered = text.lower()
            matched = [
                term for term in risk_terms
                if term in lowered
            ]

            if matched:
                risks.append({
                    "source": metadata.get("source", "Unknown"),
                    "text": text,
                    "signals": matched,
                })

        chunk_count = len(documents)
        risk_count = len(risks)

        # This is a preliminary keyword indicator, not an
        # AI prediction or a validated project health score.
        if chunk_count == 0:
            health_score = 0
        else:
            risk_ratio = risk_count / chunk_count
            health_score = max(
                0,
                min(100, 100 - round(60 * risk_ratio)),
            )

        return {
            "document_count": len(sources),
            "chunk_count": chunk_count,
            "risk_count": risk_count,
            "health_score": health_score,
            "risks": risks[:20],
        }
