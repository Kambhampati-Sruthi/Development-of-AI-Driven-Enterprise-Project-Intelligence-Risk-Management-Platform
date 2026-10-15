
def split_text(text: str, chunk_size: int = 1800,
               overlap: int = 250) -> list[str]:
    """
    Split text into character-based chunks.

    These are character counts, not token counts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a natural word boundary when possible.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(records: list[dict]) -> list[dict]:
    """Split extracted records and preserve their metadata."""
    output = []

    for record in records:
        for chunk_number, chunk in enumerate(
            split_text(record["text"]), start=1
        ):
            output.append({
                **record,
                "text": chunk,
                "chunk_number": chunk_number,
            })

    return output
