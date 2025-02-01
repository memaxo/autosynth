import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile

from autosynth.searcher import Searcher
from autosynth.collector import Collector, DocumentLoadError
from langchain.schema import Document

@pytest.mark.asyncio
async def test_search_queries_generation():
    """Test that generate_search_queries splits lines correctly."""
    searcher = Searcher(llm_model="gpt-3.5-turbo")
    
    # Mock the LLM's apredict method
    with patch.object(searcher.llm, 'apredict', return_value="Query1\nQuery2\nQuery3\n\nQuery4\nQuery5"):
        queries = await searcher.generate_search_queries("test topic")
        assert len(queries) == 5
        assert queries == ["Query1", "Query2", "Query3", "Query4", "Query5"]

@pytest.mark.asyncio
async def test_url_filtering():
    """Test that only validated URLs are kept."""
    searcher = Searcher()
    
    # Prepare mock search results
    mock_results = [
        {"link": "http://valid.com", "title": "Valid", "snippet": "valid snippet"},
        {"link": "http://invalid.com", "title": "Invalid", "snippet": "invalid snippet"}
    ]
    
    # Patch to skip the real search and to force certain validation results
    with patch.object(searcher, 'generate_search_queries', return_value=["dummy query"]), \
         patch.object(searcher.search, 'results', return_value=mock_results), \
         patch.object(searcher, '_validate_url', side_effect=[True, False]):
        final = await searcher.search_and_filter("some topic", num_results=5)
        assert len(final) == 1
        assert final[0]["url"] == "http://valid.com"

@pytest.mark.asyncio
async def test_collector_caching():
    """Test that Collector uses cached documents if available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir)
        collector = Collector(cache_dir=cache_path)
        await collector.setup()
        
        test_url = "http://example.com"
        fake_doc = Document(page_content="Hello World", metadata={})
        
        # First pass, no cache
        with patch.object(collector, '_check_rate_limit', return_value=None), \
             patch.object(collector, '_get_loader') as mock_loader, \
             patch.object(collector, '_load_with_timeout', return_value=[fake_doc]):
            docs = await collector.collect(test_url)
            assert len(docs) == 1
            mock_loader.assert_called_once()
        
        # Second pass, should retrieve from cache (mock_loader won't be called)
        with patch.object(collector, '_get_loader') as mock_loader:
            docs2 = await collector.collect(test_url)
            assert len(docs2) == 1
            mock_loader.assert_not_called()

        await collector.cleanup()

@pytest.mark.asyncio
async def test_collector_concurrency_and_rate_limit():
    """Test concurrency (semaphore) and domain-based RateLimiter usage."""
    collector = Collector(max_workers=2)
    await collector.setup()

    # Prepare multiple URLs
    urls = [f"http://test{i}.com" for i in range(5)]
    
    # Mock methods that do blocking/IO
    with patch.object(collector, '_check_rate_limit', side_effect=AsyncMock()) as mock_limit, \
         patch.object(collector, '_load_with_timeout', return_value=[]), \
         patch.object(collector, '_get_loader') as mock_loader:
        
        results = await collector.collect_batch(urls, max_concurrency=2)
        assert len(results) == 0  # we returned empty each time
        # Confirm rate limit check was called for each URL
        assert mock_limit.call_count == 5
        mock_loader.assert_called()

    await collector.cleanup()

@pytest.mark.asyncio
async def test_github_repo_loader():
    """Test that GitHub loader is triggered when the URL matches a GitHub repo."""
    collector = Collector()
    await collector.setup()

    # We'll patch out the RepoCollector usage
    with patch('autosynth.collector.RepoCollector') as MockRepoCollector:
        instance = MockRepoCollector.return_value
        instance.collect.return_value = [Document(page_content="README content", metadata={})]
        
        docs = await collector.collect("https://github.com/fake/repo")
        assert len(docs) == 1
        assert docs[0].page_content == "README content"

    await collector.cleanup()

@pytest.mark.skip(reason="Optional performance check")
@pytest.mark.asyncio
async def test_performance_sanity_check():
    """Optional: measure that collecting doesn't exceed a typical threshold."""
    collector = Collector()
    await collector.setup()

    start_time = asyncio.get_event_loop().time()
    with patch.object(collector, '_load_with_timeout', return_value=[]):
        await collector.collect("http://somefastserver.com")
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # Example threshold: 2 seconds
    assert elapsed < 2.0, f"Collection took too long: {elapsed} sec"

    await collector.cleanup()