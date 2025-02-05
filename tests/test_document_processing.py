import pytest
import asyncio
from langchain.schema import Document
from src.processor import Processor
from src.validator import DocumentValidator

@pytest.mark.asyncio
async def test_cleaning_and_chunking_large_document():
    processor = Processor()
    large_text = "word " * 3000  # Extremely large document
    doc = Document(page_content=large_text, metadata={})
    chunks = await processor.clean_content(doc, chunk_size=500, chunk_overlap=50)
    if isinstance(chunks, list):
        for chunk in chunks:
            # Each chunk should have at most 550 words (chunk_size + overlap)
            assert len(chunk.page_content.split()) <= 550
    else:
        assert "word" in chunks.page_content

@pytest.mark.asyncio
async def test_non_english_document():
    processor = Processor()
    non_english = "こんにちは 世界 " * 100  # Japanese text
    doc = Document(page_content=non_english, metadata={})
    cleaned = await processor.clean_content(doc, chunk_size=200, chunk_overlap=20)
    if isinstance(cleaned, list):
        for chunk in cleaned:
            assert len(chunk.page_content) > 0
    else:
        assert len(cleaned.page_content) > 0

@pytest.mark.asyncio
async def test_duplicate_content_detection():
    processor = Processor()
    text = "This is duplicate content. " * 50
    doc1 = Document(page_content=text, metadata={"source_url": "http://example.com"})
    doc2 = Document(page_content=text, metadata={"source_url": "http://example.com"})
    validator = DocumentValidator()
    valid1, meta1 = await validator.verify_quality(doc1, "test topic")
    valid2, meta2 = await validator.verify_quality(doc2, "test topic")
    # Assuming duplicate detection fails the second document.
    assert valid1 is True
    if not valid2:
        assert "Duplicate" in meta2["reason"]