from typing import List, Dict, Union

def chunk_text(pages_data: List[Dict[str, Union[int, str]]], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Union[int, str]]]:
    chunks = []
    chunk_index = 0

    for page in pages_data:
        page_num = page["page_number"]
        text = page["text"]

        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + chunk_size // 2:
                    end = last_space

            chunk_text_content = text[start:end].strip()

            if chunk_text_content:
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "text": chunk_text_content
                })
                chunk_index += 1

            start = end - overlap if end < text_len else text_len

    return chunks
