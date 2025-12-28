import base64
import json
from typing import Any
from openai import AsyncOpenAI
from .config import OPENAI_API_KEY, GPT_VISION_MODEL

image_vision_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


import base64
import json
from typing import Any
from openai import AsyncOpenAI
from .config import OPENAI_API_KEY, GPT_VISION_MODEL

image_vision_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def process_financial_image(img_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Process a standalone financial image (receipt, invoice, scanned statement)
    using GPT-4 Vision. Returns structured financial data.
    
    Args:
        img_bytes: Raw image bytes (PNG, JPG, etc.)
        filename: Original filename for reference
    
    Returns:
        Dictionary with extracted financial data
    """
    # Encode image to base64
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Detect image format from magic bytes
    if img_bytes[:4] == b'\x89PNG':
        mime_type = "image/png"
    elif img_bytes[:3] == b'\xff\xd8\xff':
        mime_type = "image/jpeg"
    elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default fallback
    
    prompt = """
You are a financial document OCR and extraction agent. Extract ALL financial information from this image.

Your goal is to create a CLEAR, READABLE text summary that preserves all financial details.

EXTRACT:
1. Document type (invoice, receipt, bank statement, bill, etc.)
2. Vendor/merchant name and contact info
3. Invoice/reference number
4. Dates (invoice date, due date, payment date)
5. Customer/client information
6. All line items with descriptions, quantities, unit prices, and subtotals
7. Subtotal, tax amounts, discounts
8. Total amount with currency
9. Payment terms, methods, or status
10. Any account numbers, registration numbers, or tax IDs

Create a detailed text summary with:
- Clear section headers
- Proper spacing and line breaks
- All amounts formatted clearly with currency symbols
- Complete table data if present
- Easy-to-read bullet points or numbered lists

Return STRICT JSON format:
{
  "document_type": "invoice | receipt | bank_statement | bill | financial_statement | other",
  "vendor": "vendor name or null",
  "invoice_number": "string or null",
  "date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "customer": "customer name or null",
  "currency": "USD | EUR | etc or null",
  "subtotal": 0.0 or null,
  "tax_amount": 0.0 or null,
  "discount_amount": 0.0 or null,
  "total_amount": 0.0 or null,
  "line_items": [
    {
      "description": "item name",
      "quantity": 0,
      "unit_price": 0.0,
      "amount": 0.0
    }
  ],
  "payment_method": "string or null",
  "payment_status": "PAID | UNPAID | PENDING | null",
  "reference_number": "string or null",
  "extracted_text_summary": "A well-formatted, readable summary with proper spacing, line breaks, and clear sections. Include ALL financial details from the document in an organized, easy-to-read format."
}

IMPORTANT: The "extracted_text_summary" must be well-formatted with proper spacing and structure, not a wall of text.

Do not include any text outside the JSON.
"""
    
    try:
        resp = await image_vision_client.chat.completions.create(
            model=GPT_VISION_MODEL,
            max_tokens=2000,  # Increased for better summaries
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        raw = resp.choices[0].message.content or ""
        
        # Parse JSON response
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback if GPT returns non-JSON
            data = {
                "document_type": "other",
                "vendor": None,
                "invoice_number": None,
                "date": None,
                "due_date": None,
                "customer": None,
                "currency": None,
                "subtotal": None,
                "tax_amount": None,
                "discount_amount": None,
                "total_amount": None,
                "line_items": [],
                "payment_method": None,
                "payment_status": None,
                "reference_number": None,
                "extracted_text_summary": raw[:1500]
            }
        
        # Add metadata
        data["filename"] = filename
        data["processing_status"] = "success"
        
        return data
        
    except Exception as e:
        # Error handling
        return {
            "filename": filename,
            "processing_status": "error",
            "error_message": str(e),
            "document_type": "unknown",
            "extracted_text_summary": ""
        }
