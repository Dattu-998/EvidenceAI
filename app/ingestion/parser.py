import io
import csv
import pdfplumber
from typing import List, Dict, Any, Union

def parse_pdf(file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    pages_data = []
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_data.append({
                    "page_number": i + 1,
                    "text": text
                })
    return pages_data

def parse_csv(file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    content_str = file_content.decode('utf-8')
    reader = csv.reader(io.StringIO(content_str))

    rows_data = []
    header = next(reader, None)

    for i, row in enumerate(reader):
        row_text = ""
        if header:
            row_text = ", ".join([f"{header[j]}: {val}" for j, val in enumerate(row) if j < len(header)])
        else:
            row_text = ", ".join(row)

        if row_text:
            rows_data.append({
                "page_number": i + 1,
                "text": row_text
            })

    return rows_data

def parse_file(filename: str, file_content: bytes) -> List[Dict[str, Union[int, str]]]:
    if filename.lower().endswith('.pdf'):
        return parse_pdf(file_content)
    elif filename.lower().endswith('.csv'):
        return parse_csv(file_content)
    else:
        raise ValueError(f"Unsupported file format for {filename}")
