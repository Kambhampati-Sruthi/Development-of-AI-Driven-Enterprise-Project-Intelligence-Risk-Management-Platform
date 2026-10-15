
from app.chunking import create_chunks, split_text


def test_empty_text():
    assert split_text("") == []


def test_short_text():
    assert split_text("Sprint planning completed.") == [
        "Sprint planning completed."
    ]


def test_chunk_size():
    text = " ".join(["project"] * 500)

    chunks = split_text(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_metadata_preserved():
    records = [{
        "text": "The API task is blocked.",
        "source": "sprint.csv",
        "file_type": "csv",
        "page": None,
        "row": 4,
    }]

    chunks = create_chunks(records)

    assert chunks[0]["source"] == "sprint.csv"
    assert chunks[0]["row"] == 4
    assert chunks[0]["chunk_number"] == 1
