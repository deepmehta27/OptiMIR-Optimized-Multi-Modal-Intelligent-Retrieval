from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .image_ingest import process_financial_image
from .ingest import get_or_create_collection, UPLOADED_SOURCES

router = APIRouter()


@router.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    """
    Ingest financial images (receipts, invoices, scanned documents)
    using GPT-4 Vision for OCR and data extraction.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only image files (JPG, PNG, WEBP) are supported"
        )
    
    # Read image bytes
    img_bytes = await file.read()
    
    # Process with GPT-4 Vision
    result = await process_financial_image(img_bytes, filename=file.filename)
    
    if result["processing_status"] == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {result.get('error_message', 'Unknown error')}"
        )
    
    # Extract the text summary for indexing
    extracted_text = result.get("extracted_text_summary", "")
    
    if not extracted_text or len(extracted_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Could not extract meaningful financial data from this image"
        )
    
    # Format line items
    line_items_text = ""
    if result.get("line_items"):
        line_items_text = "\n\nLine Items:"
        for idx, item in enumerate(result["line_items"], 1):
            line_items_text += f"\n  {idx}. {item.get('description', 'N/A')}"
            if item.get('quantity'):
                line_items_text += f" - Qty: {item['quantity']}"
            if item.get('unit_price'):
                line_items_text += f" @ {result.get('currency', '')} {item['unit_price']}"
            if item.get('amount'):
                line_items_text += f" = {result.get('currency', '')} {item['amount']}"
    
    # Create a searchable document from the extracted data
    doc_text = f"""
=== FINANCIAL DOCUMENT ===

Document Type: {result.get('document_type', 'Unknown').upper()}
Source: {file.filename}

--- VENDOR INFORMATION ---
Vendor: {result.get('vendor', 'N/A')}

--- CLIENT INFORMATION ---
Customer: {result.get('customer', 'N/A')}

--- DOCUMENT DETAILS ---
Invoice Number: {result.get('invoice_number', 'N/A')}
Reference: {result.get('reference_number', 'N/A')}
Invoice Date: {result.get('date', 'N/A')}
Due Date: {result.get('due_date', 'N/A')}
Payment Status: {result.get('payment_status', 'N/A')}
Payment Method: {result.get('payment_method', 'N/A')}

--- FINANCIAL SUMMARY ---
Subtotal: {result.get('currency', '')} {result.get('subtotal', 'N/A')}
Tax: {result.get('currency', '')} {result.get('tax_amount', 'N/A')}
Discount: {result.get('currency', '')} {result.get('discount_amount', 'N/A') if result.get('discount_amount') else 'None'}
TOTAL AMOUNT: {result.get('currency', '')} {result.get('total_amount', 'N/A')}

{line_items_text}

--- EXTRACTED DETAILS ---
{extracted_text}
"""
    
    # Chunk and store in ChromaDB
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    
    chunks = splitter.split_text(doc_text)
    
    collection = get_or_create_collection()
    
    # Create metadata
    all_metadata = []
    for idx in range(len(chunks)):
        # Base metadata (always present)
        meta = {
            "source": file.filename,
            "page": 1,
            "type": "image",
            "chunk_index": idx,
        }
        
        # ✅ Only add optional fields if they're not None
        if result.get("document_type"):
            meta["document_type"] = result["document_type"]
        if result.get("vendor"):
            meta["vendor"] = result["vendor"]
        if result.get("date"):
            meta["date"] = result["date"]
        if result.get("total_amount"):
            meta["total_amount"] = result["total_amount"]
        
        all_metadata.append(meta)
    
    ids = [f"{file.filename}_img_c{m['chunk_index']}" for m in all_metadata]
    
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=all_metadata,
    )
    
    # Add to uploaded sources
    UPLOADED_SOURCES.add(file.filename)
    
    return {
        "status": "success",
        "filename": file.filename,
        "document_type": result.get("document_type"),
        "chunks_created": len(chunks),
        "extracted_data": {
            "vendor": result.get("vendor"),
            "date": result.get("date"),
            "total": result.get("total_amount"),
        }
    }
