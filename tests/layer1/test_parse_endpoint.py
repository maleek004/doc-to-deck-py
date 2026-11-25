"""Integration tests: the whole app (dependencies, routing, exception handlers, JSON) runs in
memory and is called over HTTP, as the browser would. TestClient opens no port."""

import pytest
from fastapi.testclient import TestClient

from doctodeck.app import NO_FILE, create_app, get_parser_options
from doctodeck.errors import DocumentErrorCode
from doctodeck.parser import ParserOptions
from tests.pdf_builder import build_pdf


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def upload(client: TestClient, data: bytes, filename: str):
    return client.post("/api/parse", files={"file": (filename, data)})  # field name must be "file"


def test_markdown_upload_returns_the_outline(client):
    response = upload(client, b"# Async\nCoroutines.\n# Cancel\nCancelledError.", "notes.md")

    assert response.status_code == 200
    body = response.json()
    assert body["fileName"] == "notes.md"
    assert [s["title"] for s in body["document"]["sections"]] == ["Async", "Cancel"]


def test_pdf_upload_returns_the_outline_and_page_count(client):
    pdf = build_pdf(["1. Arrays", "Hold items."], ["2. Lists", "Can grow."])

    response = upload(client, pdf, "notes.pdf")

    assert response.status_code == 200
    document = response.json()["document"]
    assert document["pageCount"] == 2
    assert [s["title"] for s in document["sections"]] == ["1. Arrays", "2. Lists"]


def test_json_uses_camel_case_property_names(client):
    body = upload(client, b"# A\nb", "notes.md").json()

    # The page's JavaScript reads body.document.sections[0].wordCount, so these names are a contract.
    assert set(body) == {"fileName", "document"}
    assert set(body["document"]) == {"sections", "cleanText", "wordCount", "pageCount"}
    assert set(body["document"]["sections"][0]) == {"title", "level", "wordCount", "snippet"}


def test_missing_file_is_a_400_with_code_no_file(client):
    response = client.post("/api/parse", data={"note": "no file here"})

    assert response.status_code == 400
    assert response.json()["error"] == NO_FILE


def test_unsupported_file_type_is_a_415(client):
    response = upload(client, b"hello", "notes.docx")

    assert response.status_code == 415
    assert response.json()["error"] == DocumentErrorCode.UNSUPPORTED_FILE_TYPE


def test_fake_pdf_is_a_400_with_its_error_code(client):
    response = upload(client, b"not a pdf", "fake.pdf")

    assert response.status_code == 400
    assert response.json() == {
        "error": "INVALID_FILE_SIGNATURE",
        "message": "The file is named .pdf but its contents are not a PDF.",
    }


def test_file_over_the_limit_is_a_413(app, client):
    # Swap in tiny limits for this test only: FastAPI calls the override instead of the real dependency.
    app.dependency_overrides[get_parser_options] = lambda: ParserOptions(max_file_size_bytes=10)

    response = upload(client, b"# Way more than ten bytes", "notes.md")

    assert response.status_code == 413
    assert response.json()["error"] == DocumentErrorCode.FILE_TOO_LARGE


def test_home_page_serves_the_upload_form(client):
    html = client.get("/").text

    assert 'id="parse-form"' in html
