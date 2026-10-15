
import hashlib
from pathlib import Path

import chromadb

from app.embeddings import EmbeddingService


DEFAULT_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "project_documents"


class ProjectVectorStore:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        db_path: str = DEFAULT_DB_PATH,
    ):
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.embedding_service = embedding_service

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedding_service.embed_documents(texts)

        ids = []
        metadatas = []

        for chunk, vector in zip(chunks, vectors):
            identity = "|".join([
                chunk["source"],
                str(chunk.get("page")),
                str(chunk.get("row")),
                str(chunk.get("chunk_number")),
                chunk["text"],
            ])

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

            metadata = {
                "source": str(chunk["source"]),
                "file_type": str(chunk["file_type"]),
                "page": int(chunk["page"] or 0),
                "row": str(chunk["row"] or ""),
                "chunk_number": int(chunk["chunk_number"]),
            }

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

        return len(ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.collection.count() == 0:
            return []

        query_vector = self.embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append({
                "text": document,
                "source": metadata["source"],
                "file_type": metadata["file_type"],
                "page": metadata["page"] or None,
                "row": metadata["row"] or None,
                "chunk_number": metadata["chunk_number"],
                "distance": float(distance),
            })

        return matches

    def clear(self) -> None:
        """Remove all chunks from the current collection."""
        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
