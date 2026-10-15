
from io import BytesIO

import pandas as pd
import pytest
from docx import Document

from app.ingestion import extract_document


def test_txt_ingestion():
    records = extract_document(
        "notes.txt",
        b"Sprint 1 has two incomplete tasks.",
    )

    assert len(records) == 1
    assert "two incomplete tasks" in records[0]["text"]
    assert records[0]["source"] == "notes.txt"


def test_csv_ingestion():
    frame = pd.DataFrame({
        "Task": ["API integration"],
        "Status": ["Blocked"],
    })

    buffer = BytesIO()
    frame.to_csv(buffer, index=False)

    records = extract_document(
        "tasks.csv",
        buffer.getvalue(),
    )

    assert len(records) == 1
    assert "API integration" in records[0]["text"]
    assert "Blocked" in records[0]["text"]
    assert records[0]["row"] == 2


def test_docx_ingestion():
    document = Document()
    document.add_paragraph("Sprint review is scheduled for Friday.")

    buffer = BytesIO()
    document.save(buffer)

    records = extract_document(
        "meeting.docx",
        buffer.getvalue(),
    )

    assert any(
        "Sprint review" in record["text"]
        for record in records
    )


def test_unsupported_file():
    with pytest.raises(ValueError):
        extract_document("notes.exe", b"test")


def test_empty_file():
    with pytest.raises(ValueError):
        extract_document("notes.txt", b"")
