import pytest
import asyncio
import json
from pathlib import Path
from src.db import ProjectStateDB, AutoSynthDB
import aiosqlite

@pytest.fixture
async def temp_db(tmp_path):
    db_path = tmp_path / "project.db"
    db = ProjectStateDB(db_type="project")
    db.db_file = str(db_path)
    await db._init_tables()
    yield db

@pytest.mark.asyncio
async def test_create_and_get_project(temp_db):
    project_id = "test_proj"
    topic = "testing"
    config = {"key": "value"}
    await temp_db.create_project(project_id, topic, config)
    proj = await temp_db.get_project(project_id)
    assert proj is not None
    assert proj["project_id"] == project_id
    assert proj["topic"] == topic
    assert proj["config"] == config

@pytest.mark.asyncio
async def test_duplicate_url_insertion(temp_db):
    project_id = "test_proj"
    urls = ["http://example.com", "http://example.com"]
    await temp_db.add_urls(project_id, urls)
    retrieved = await temp_db.get_urls(project_id)
    assert len(retrieved) == 1

@pytest.mark.asyncio
async def test_update_metrics(temp_db):
    project_id = "test_proj"
    await temp_db.create_project(project_id, "testing", {})
    await temp_db.update_metrics(project_id, "tokens", 100, 200)
    metrics = await temp_db.get_metrics(project_id)
    assert "tokens" in metrics
    assert metrics["tokens"]["current"] == 100
    assert metrics["tokens"]["target"] == 200

@pytest.mark.asyncio
async def test_query_timeout_simulation(temp_db):
    import asyncio
    from unittest.mock import patch
    async def delayed_connect(*args, **kwargs):
        await asyncio.sleep(0.1)
        return await aiosqlite.connect(*args, **kwargs)
    with patch("aiosqlite.connect", side_effect=delayed_connect):
        project_id = "test_proj"
        await temp_db.create_project(project_id, "testing", {})
        proj = await temp_db.get_project(project_id)
        assert proj is not None

@pytest.mark.asyncio
async def test_corrupted_cache_file(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "12345_repo.pkl"
    cache_file.write_text("corrupted content")
    from src.collector import Collector
    collector = Collector(cache_dir=cache_dir)
    result = await collector._check_cache("dummy_url")
    assert result is None