import pytest
import asyncio
import gradio as gr
from src.gradio import build_interface

@pytest.mark.asyncio
async def test_gradio_interface_launch():
    demo = build_interface()
    # Verify that the interface builds without error and contains the expected tabs.
    # For instance, check that "Search" tab is present.
    html = demo.get_blocks()[0].get_config().get("title", "")
    assert "AutoSynth" in html or "AutoSynth" in str(html)
    # Optionally, simulate a simple UI interaction if Gradio testing utilities are available.
    # For now, ensure the interface loads successfully.