import pytest
import asyncio
from unittest.mock import patch, MagicMock
import os
import tempfile
import pickle
from pathlib import Path

from autosynth.generator import ContentGenerator
from autosynth.pipeline import AutoSynth, Project
from langchain.schema import Document

@pytest.mark.asyncio
async def test_basic_generation():
    """Test the ContentGenerator for basic generation behavior."""
    generator = ContentGenerator()
    docs = [Document(page_content="Original content", metadata={})]
    
    # Mock the underlying Curator/LLM call
    with patch.object(generator, '__call__', return_value=[{
        "original": "Original content",
        "generated_content": "New content variation",
        "preservation_analysis": "Key points preserved",
        "metadata": {}
    }]):
        results = await generator.generate_batch(docs)
        assert len(results) == 1
        assert results[0]["generated_content"] == "New content variation"

@pytest.mark.asyncio
async def test_pipeline_generate_batch_checkpoint():
    """Test that the pipeline's generate stage respects batch_size and checkpoints."""
    
    # Create an in-memory AutoSynth instance with minimal config
    with tempfile.TemporaryDirectory() as tmpdir:
        autosynth = AutoSynth(base_path=tmpdir)
        project = autosynth.create_project("Test Topic")
        
        # Simulate processed_docs in project state
        project.state.processed_docs = [
            {"content": f"Doc {i}", "metadata": {}} for i in range(10)
        ]
        project.state.stage_configs["generate"]["batch_size"] = 3
        project.state.stage_configs["generate"]["max_retries"] = 2
        
        # Force the generator to produce a known result
        mock_result = {
            "original": "",
            "generated_content": "Generated",
            "preservation_analysis": "",
            "metadata": {}
        }
        
        # Patch the actual generator call
        with patch.object(project.generator, 'generate_batch', return_value=[mock_result, mock_result]) as mock_gen:
            # Run stage with a small target
            await autosynth._run_generate(project, target_variations=5)
            
            # Check that we do partial batches and checkpoint
            # e.g. first batch yields 2 docs, second yields 2, etc. until we have 5 or run out.
            assert len(project.state.generated_docs) == 5
            assert project.state.checkpoints["generate_batch"] > 0  # e.g. 6 or 9

@pytest.mark.asyncio
async def test_pipeline_generate_retry():
    """Test that the pipeline generation retries on API error, and partial success is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        autosynth = AutoSynth(base_path=tmpdir)
        project = autosynth.create_project("Another topic")
        
        # 3 docs to generate
        project.state.processed_docs = [
            {"content": "doc1", "metadata": {}},
            {"content": "doc2", "metadata": {}},
            {"content": "doc3", "metadata": {}},
        ]
        project.state.stage_configs["generate"]["batch_size"] = 3
        project.state.stage_configs["generate"]["max_retries"] = 2
        
        # Make the generator fail on first attempt, succeed on second
        fail_once = [True]  # list as mutable closure
        
        async def mock_generate_batch(docs):
            if fail_once[0]:
                fail_once[0] = False
                raise Exception("Simulated API error")
            return [{
                "original": d.page_content,
                "generated_content": "OK",
                "preservation_analysis": "None",
                "metadata": {}
            } for d in docs]
        
        with patch.object(project.generator, 'generate_batch', side_effect=mock_generate_batch):
            await autosynth._run_generate(project, target_variations=3)
            # Should succeed on second attempt, so 3 docs generated
            assert len(project.state.generated_docs) == 3
            # The pipeline should have saved partial results or retried correctly
            # Confirm we didn't raise an error or skip the final results

@pytest.mark.asyncio
async def test_pipeline_run_stage_e2e():
    """Full stage run example with minimal discovered URLs and a mock collector/generator."""
    with tempfile.TemporaryDirectory() as tmpdir:
        autosynth = AutoSynth(base_path=tmpdir)
        project = autosynth.create_project("Test E2E")

        # Stage 1: discover
        # Fake discovered_urls
        project.state.discovered_urls = ["http://mock1.com", "http://mock2.com"]
        project.state.metrics["target_urls"] = 2
        # skip searching to keep test short

        # Stage 2: collect
        with patch.object(project.collector, 'collect', return_value=[Document(page_content="DocContent", metadata={})]):
            await autosynth._run_collect(project, target_tokens=5)
            assert len(project.state.collected_docs) == 2, "Should have collected from both URLs"

        # Stage 3: process
        # Fake processor, just replicate doc as 'processed'
        with patch.object(project.processor, 'clean_content', side_effect=lambda d, chunk_size, chunk_overlap: d), \
             patch.object(project.processor, 'verify_quality', return_value=True):
            await autosynth._run_process(project, target_chunks=2)
            assert len(project.state.processed_docs) == 2

        # Stage 4: generate
        with patch.object(project.generator, 'generate_batch', return_value=[{
            "original": "DocContent",
            "generated_content": "Synthesized data",
            "preservation_analysis": "Key points",
            "metadata": {}
        }]):
            await autosynth._run_generate(project, target_variations=2)
            assert len(project.state.generated_docs) == 2

        # Confirm pipeline updated
        assert project.state.last_stage == "generate"

@pytest.mark.asyncio
async def test_generator_batch():
    """Test batch generation of variations."""
    project = Project("test", "python", Path("test_data"))
    
    # Create test documents
    docs = [
        Document(page_content="Test content 1", metadata={"source": "test1"}),
        Document(page_content="Test content 2", metadata={"source": "test2"})
    ]
    
    # Add test documents to database
    for doc in docs:
        project.state_db.add_processed_doc(
            project.project_id,
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        )
    
    # Run generation
    await project._run_generate(5)
    
    # Check results
    generated_docs = project.state.get_generated_docs(project.state_db)
    assert len(generated_docs) == 5
    
    # Verify content
    for doc in generated_docs:
        assert isinstance(doc["content"], str)
        assert len(doc["content"]) > 0
        assert "original" in doc
        assert "analysis" in doc
        assert "metadata" in doc

@pytest.mark.asyncio
async def test_generator_with_invalid_docs():
    """Test generation with some invalid documents."""
    project = Project("test2", "python", Path("test_data"))
    
    # Create test documents (some invalid)
    docs = [
        Document(page_content="Valid content", metadata={"source": "test1"}),
        Document(page_content="", metadata={"source": "test2"}),  # Invalid
        Document(page_content="Another valid", metadata={"source": "test3"})
    ]
    
    # Add test documents to database
    for doc in docs:
        project.state_db.add_processed_doc(
            project.project_id,
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        )
    
    # Run generation
    await project._run_generate(3)
    
    # Check results
    generated_docs = project.state.get_generated_docs(project.state_db)
    assert len(generated_docs) == 3
    
    # Verify content
    for doc in generated_docs:
        assert len(doc["content"]) > 0

@pytest.mark.asyncio
async def test_full_pipeline():
    """Test the complete pipeline flow."""
    project = Project("test3", "python testing", Path("test_data"))
    
    # Test URLs
    test_urls = [
        "https://example.com/test1",
        "https://example.com/test2"
    ]
    project.state.discovered_urls = test_urls
    
    # Run collection
    await project._run_collect(1000)
    collected_docs = project.state.get_collected_docs(project.state_db)
    assert len(collected_docs) == 2, "Should have collected from both URLs"
    
    # Run processing
    await project._run_process(5)
    processed_docs = project.state.get_processed_docs(project.state_db)
    assert len(processed_docs) == 2
    
    # Run generation
    await project._run_generate(2)
    generated_docs = project.state.get_generated_docs(project.state_db)
    assert len(generated_docs) == 2
    
    # Verify metrics were updated
    assert project.state.metrics.current_tokens > 0
    assert project.state.metrics.current_chunks > 0
    assert project.state.metrics.current_variations > 0