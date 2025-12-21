import base64
import json
from typing import Any
import fitz  
import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_VISION_MODEL
vision_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

def page_needs_vision(page: fitz.Page) -> bool:
    """
    Decide if this page is worth sending to Claude vision.
    Cheap checks: images, tables, very blocky layout.
    """
    # 1) Any embedded images?
    if page.get_images():
        return True

    # 2) Try built-in table detection (if available)
    try:
        tabs = page.find_tables()
        if getattr(tabs, "tables", []):
            return True
    except Exception:
        pass

    # 3) Block-density heuristic
    blocks = page.get_text("blocks")
    plain = page.get_text("text") or ""
    if len(blocks) > 20 and len(plain.strip()) < 200:
        return True

    return False

async def summarize_image_with_vision(img_bytes: bytes, page_num: int) -> dict[str, Any]:
    """
    Call Claude vision on a PNG image of a page, return parsed JSON summary.
    """
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "You are an OCR + document understanding agent.\n"
        "You are given a single PDF page as an image.\n\n"
        "TASKS:\n"
        "1. Identify any tables or charts.\n"
        "2. If a table exists, reconstruct it as a markdown table.\n"
        "3. Summarize the most important numeric or categorical relationships.\n"
        "4. If there is no table, summarize the main points.\n\n"
        "Output a JSON object ONLY, with fields:\n"
        '{\n'
        '  \"page\": <page_number>,\n'
        '  \"has_table\": true/false,\n'
        '  \"summary\": \"natural language summary\",\n'
        '  \"table_markdown\": \"markdown table or empty string\"\n'
        '}\n'
        "Do not add any other text."
    )

    resp = await vision_client.messages.create(
        model=CLAUDE_VISION_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    text_parts = [b.text for b in resp.content if b.type == "text"]
    raw = "".join(text_parts).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "page": page_num,
            "has_table": False,
            "summary": raw,
            "table_markdown": "",
        }

    if "page" not in data:
        data["page"] = page_num
    return data