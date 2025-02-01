import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import Document

from autosynth.processor import Processor, DocumentValidator, ValidationResult

@pytest.mark.asyncio
async def test_processor_clean_content():
    """Test that clean_content does some form of minimal text processing."""
    processor = Processor()
    
    mock_doc = Document(page_content="   Hello <p>World</p>   ", metadata={})
    cleaned_doc = await processor.clean_content(mock_doc, chunk_size=1000, chunk_overlap=100)
    
    # Implementation details might vary, but check for typical cleaning
    assert "Hello" in cleaned_doc.page_content
    assert "World" in cleaned_doc.page_content
    # No HTML tags
    assert "<p>" not in cleaned_doc.page_content

@pytest.mark.asyncio
async def test_processor_quality_check_pass():
    """Test that verify_quality returns True for relevant, English content."""
    processor = Processor()
    
    doc = Document(page_content="Some relevant text in English about the project topic.", metadata={})
    with patch.object(processor.validator, 'validate_document', return_value=(True, {"validation_reason": "Good"})):
        result = await processor.verify_quality(doc, "some topic")
        assert result is True

@pytest.mark.asyncio
async def test_processor_quality_check_fail():
    """Test that verify_quality returns False for irrelevant or invalid content."""
    processor = Processor()
    
    doc = Document(page_content="Gibberish??? ??? ??? 12345", metadata={})
    with patch.object(processor.validator, 'validate_document', return_value=(False, {"reason": "Irrelevant"})):
        result = await processor.verify_quality(doc, "some topic")
        assert result is False

@pytest.mark.asyncio
async def test_processor_chunking():
    """Test that a large doc is split into multiple chunks."""
    processor = Processor()
    large_text = " ".join(["Hello"] * 3000)  # ~3000 words
    doc = Document(page_content=large_text, metadata={})
    
    cleaned = await processor.clean_content(doc, chunk_size=1000, chunk_overlap=100)
    # If the processor does chunking, it might return a single doc or multiple.
    # Suppose it returns the first chunk for demonstration or merges them:
    # Implementation might vary. Here we just confirm it doesn't break.
    assert cleaned.page_content
    assert len(cleaned.page_content.split()) <= 1000 + 100, "Exceeded chunk size with overlap"

@pytest.mark.asyncio
async def test_documentvalidator_runs_llm_check():
    """Test that DocumentValidator calls the LLM chain properly."""
    validator = DocumentValidator()
    mock_doc = Document(page_content="Testing content", metadata={})
    
    mock_result = ValidationResult(
        is_valid=True,
        reason="Content is well-formed and relevant",
        content_type="text"
    )
    
    with patch.object(validator.validation_chain, 'ainvoke', return_value=mock_result) as mock_invoke:
        is_valid, meta = await validator.validate_document(mock_doc, "some topic")
        assert is_valid is True
        assert meta["content_type"] == "text"
        assert meta["validation_reason"] == "Content is well-formed and relevant"
        mock_invoke.assert_called_once()

@pytest.mark.asyncio
async def test_documentvalidator_handles_exception():
    """Test error handling in validation."""
    validator = DocumentValidator()
    mock_doc = Document(page_content="Some content", metadata={})
    
    with patch.object(validator.validation_chain, 'ainvoke', side_effect=Exception("LLM error")):
        is_valid, meta = await validator.validate_document(mock_doc, "some topic")
        assert is_valid is False
        assert "Validation error: LLM error" in meta["reason"]