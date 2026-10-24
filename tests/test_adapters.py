import json

import pymupdf

from dataset.adapters.json_adapter import load_json_documents
from dataset.adapters.csv_adapter import load_csv_documents
from dataset.adapters.pdf_adapter import load_pdf_documents


def test_json_adapter(tmp_path):
    file_path = tmp_path / "test.json"

    data = {
        "title": "Payment Incident",
        "content": "Payment service failed.",
        "severity": "high",
    }

    file_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    documents = load_json_documents(file_path)

    assert len(documents) == 1
    assert documents[0].title == "Payment Incident"
    assert documents[0].content == "Payment service failed."
    assert documents[0].metadata["severity"] == "high"
    assert documents[0].metadata["filename"] == "test.json"


def test_jsonl_adapter(tmp_path):
    file_path = tmp_path / "test.jsonl"

    records = [
        {
            "title": "First Incident",
            "content": "Database connection failed.",
        },
        {
            "title": "Second Incident",
            "content": "Authentication service failed.",
        },
    ]

    file_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    documents = load_json_documents(file_path)

    assert len(documents) == 2
    assert documents[0].title == "First Incident"
    assert documents[1].title == "Second Incident"
    assert documents[0].content == "Database connection failed."
    assert documents[1].content == "Authentication service failed."


def test_csv_adapter(tmp_path):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "title,content,severity\n"
        "Payment Incident,Payment service failed.,high\n"
        "Login Incident,Authentication service failed.,medium\n",
        encoding="utf-8",
    )

    documents = load_csv_documents(file_path)

    assert len(documents) == 2
    assert documents[0].title == "Payment Incident"
    assert documents[1].title == "Login Incident"
    assert documents[0].metadata["severity"] == "high"
    assert documents[1].metadata["severity"] == "medium"
    assert documents[0].metadata["row_number"] == 2
    assert documents[1].metadata["row_number"] == 3


def test_pdf_adapter(tmp_path):
    file_path = tmp_path / "test.pdf"

    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text(
            (72, 72),
            "Payment service failed.",
        )

        second_page = pdf.new_page()
        second_page.insert_text(
            (72, 72),
            "Database migration caused the incident.",
        )

        pdf.save(file_path)

    documents = load_pdf_documents(file_path)

    assert len(documents) == 2

    assert "Payment service failed." in documents[0].content
    assert "Database migration caused the incident." in documents[1].content

    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2

    assert documents[0].metadata["total_pages"] == 2
    assert documents[1].metadata["total_pages"] == 2