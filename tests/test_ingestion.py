import pytest
from app.ingestion.parser import parse_csv, parse_pdf
from app.ingestion.chunker import chunk_text
import io
from reportlab.pdfgen import canvas

def test_parse_csv():
    csv_content = b"header1,header2\nval1,val2\nval3,val4"
    data = parse_csv(csv_content)
    assert len(data) == 2
    assert data[0]["text"] == "header1: val1, header2: val2"
    assert data[0]["page_number"] == 1
    assert data[1]["text"] == "header1: val3, header2: val4"
    assert data[1]["page_number"] == 2

def test_parse_pdf():
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "Page 1 content")
    c.showPage()
    c.drawString(100, 750, "Page 2 content")
    c.showPage()
    c.save()

    data = parse_pdf(pdf_buffer.getvalue())
    assert len(data) == 2
    assert "Page 1 content" in data[0]["text"]
    assert data[0]["page_number"] == 1
    assert "Page 2 content" in data[1]["text"]
    assert data[1]["page_number"] == 2

def test_chunk_text():
    pages_data = [
        {"page_number": 1, "text": "A" * 1500}
    ]
    chunks = chunk_text(pages_data, chunk_size=1000, overlap=100)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert len(chunks[0]["text"]) == 1000
    assert chunks[1]["page_number"] == 1
    assert len(chunks[1]["text"]) == 600
