
from app.rag_pipeline import RAGPipeline


class ProjectIntelligenceAssistant:
    """Answer project questions using retrieved project-document context."""

    def __init__(self, rag_pipeline=None, top_k=5):
        self.rag = rag_pipeline or RAGPipeline()
        self.top_k = top_k

    @staticmethod
    def _get_value(item, *keys, default=""):
        """Read a field from a dictionary without assuming its schema."""
        if not isinstance(item, dict):
            return default

        for key in keys:
            value = item.get(key)
            if value is not None and value != "":
                return value

        return default

    def ask(self, question):
        """
        Retrieve project context and return an answer with source references.

        This version deliberately does not claim to be a generative LLM.
        It returns document-grounded excerpts without an external API.
        """
        question = (question or "").strip()

        if not question:
            return {
                "question": question,
                "answer": "Please enter a project-related question.",
                "sources": [],
                "grounded": False,
            }

        try:
            results = self.rag.search(
                query=question,
                n_results=self.top_k,
            )
        except Exception as exc:
            return {
                "question": question,
                "answer": (
                    "I could not search the project knowledge base. "
                    "Check that the RAG pipeline and local embedding "
                    f"model are available. Details: {exc}"
                ),
                "sources": [],
                "grounded": False,
            }

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        # Chroma may return nested lists for a single query.
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]

        sources = []
        seen = set()

        for index, document in enumerate(documents):
            if not document or not str(document).strip():
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas) and isinstance(
                    metadatas[index], dict
                )
                else {}
            )

            filename = self._get_value(
                metadata,
                "filename",
                "source",
                "file_name",
                default="Unknown source",
            )

            chunk_id = self._get_value(
                metadata,
                "chunk_index",
                "chunk_id",
                default=str(index + 1),
            )

            identity = (str(filename), str(chunk_id))
            if identity in seen:
                continue
            seen.add(identity)

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            sources.append({
                "filename": str(filename),
                "chunk_id": str(chunk_id),
                "content": str(document).strip(),
                "distance": distance,
            })

        if not sources:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant passages in the indexed "
                    "project documents. Upload and index project files, "
                    "then try again."
                ),
                "sources": [],
                "grounded": False,
            }

        excerpts = []
        for index, source in enumerate(sources, start=1):
            excerpts.append(
                f"[Source {index}: {source['filename']}]\n"
                f"{source['content']}"
            )

        answer = (
            "Relevant information found in the project documents:\n\n"
            + "\n\n".join(excerpts)
            + "\n\nThese are retrieved document excerpts, not an "
              "independently verified conclusion. Review the cited "
              "sources before making project decisions."
        )

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "filename": source["filename"],
                    "chunk_id": source["chunk_id"],
                }
                for source in sources
            ],
            "grounded": True,
        }
