import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from langchain.schema import Document
from src.validator import DocumentValidator
from src.providers.brave import BraveSearchAPIWrapper

@pytest.mark.asyncio
async def test_llm_failure_in_validator():
    validator = DocumentValidator()
    doc = Document(page_content="Test content", metadata={"source_url": "http://example.com"})
    with patch.object(validator.validation_chain, 'ainvoke', side_effect=Exception("LLM failure")):
        is_valid, meta = await validator.verify_quality(doc, "test topic")
        assert is_valid is False
        assert "LLM validation failed" in meta["reason"]

@pytest.mark.asyncio
async def test_rate_limit_exception_in_provider():
    provider = BraveSearchAPIWrapper(api_key="dummy")
    with patch.object(provider, "_acquire_rate_limit", side_effect=Exception("Rate limit exceeded")):
        with pytest.raises(Exception) as excinfo:
            await provider.results("test query", num_results=1)
        assert "Rate limit exceeded" in str(excinfo.value)

@pytest.mark.asyncio
async def test_invalid_url_handling_in_validator():
    validator = DocumentValidator()
    doc = Document(page_content="Some content", metadata={"source_url": ""})
    is_valid, meta = await validator.verify_quality(doc, "test topic")
    assert is_valid is False
    assert "No URL provided" in meta["reason"]