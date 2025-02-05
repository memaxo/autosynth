import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from langchain.schema import Document

from src.providers.brave import BraveSearchAPIWrapper
from src.providers.duckduckgo import DuckDuckGoSearchAPIWrapper
from src.providers.exa import ExaAPIWrapper
from src.providers.google import GoogleSearchAPIWrapper
from src.providers.tavily import TavilySearchAPIWrapper

# Dummy JSON response to simulate API responses
dummy_json = '[{"link": "http://example.com", "title": "Example", "snippet": "Example snippet"}]'

@pytest.mark.asyncio
async def test_brave_provider_success():
    provider = BraveSearchAPIWrapper(api_key="dummy")
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider.tool, "run", return_value=dummy_json) as mock_run:
        results = await provider.results("test query", num_results=1)
        assert isinstance(results, list)
        for res in results:
            assert "link" in res
            assert "title" in res
            assert "snippet" in res

@pytest.mark.asyncio
async def test_brave_provider_error():
    provider = BraveSearchAPIWrapper(api_key="dummy")
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider.tool, "run", side_effect=Exception("Timeout")) as mock_run, \
         patch.object(provider, "_handle_error") as mock_handle_error:
        with pytest.raises(Exception):
            await provider.results("test query", num_results=1)
        mock_handle_error.assert_called_once()

@pytest.mark.asyncio
async def test_duckduckgo_provider_success():
    provider = DuckDuckGoSearchAPIWrapper()
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider.tool, "invoke", return_value='[{"link": "http://ddg.com", "title": "DuckDuckGo", "snippet": "Snippet"}]') as mock_invoke:
        results = await provider.results("query", num_results=1)
        assert isinstance(results, list)
        for res in results:
            assert "link" in res
            assert "title" in res
            assert "snippet" in res

@pytest.mark.asyncio
async def test_exa_provider_success():
    provider = ExaAPIWrapper(api_key="dummy")
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider, "_execute_search", return_value=[{"url": "http://exa.com", "title": "Exa", "text": "Snippet"}]) as mock_exec:
        results = await provider.results("query", num_results=1)
        assert isinstance(results, list)
        # The Exa provider processes items into a standard format.
        for res in results:
            # Either "link" or "url" should be available in the result.
            assert "link" in res or "url" in res

@pytest.mark.asyncio
async def test_google_provider_success():
    provider = GoogleSearchAPIWrapper(api_key="dummy", cse_id="dummy")
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider.search_wrapper, "results", return_value=[{"link": "http://google.com", "title": "Google", "snippet": "Snippet", "fileFormat": ""}]) as mock_results:
        results = await provider.results("query", num_results=1, academic_only=True)
        assert isinstance(results, list)
        for res in results:
            assert "link" in res
            assert "title" in res
            assert "snippet" in res

@pytest.mark.asyncio
async def test_tavily_provider_success():
    provider = TavilySearchAPIWrapper(api_key="dummy")
    # Simulate a response object with an 'artifact' attribute.
    mock_response = MagicMock()
    mock_response.artifact = {"results": [{"url": "http://tavily.com", "title": "Tavily", "content": "Snippet"}]}
    with patch.object(provider, "_acquire_rate_limit", new=AsyncMock()) as mock_rate, \
         patch.object(provider.tool, "invoke", return_value=mock_response) as mock_invoke:
        results = await provider.results("query", num_results=1)
        assert isinstance(results, list)
        for res in results:
            # Depending on processing, either "link" or "url" key might be present.
            assert ("link" in res or "url" in res)
            assert "title" in res
            assert "snippet" in res