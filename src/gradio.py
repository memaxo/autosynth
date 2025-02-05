"""
Gradio web interface for AutoSynth.

This module provides a web-based interface for:
- Document search and collection
- Content validation and processing
- Synthetic content generation
- Pipeline monitoring and visualization
"""

import asyncio
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

from .collector import Collector
from .model import PhiValidator
from .monitor import Monitor, create_progress_bar
from .processor import DocumentValidator, Processor
from .providers.brave import BraveAPIWrapper
from .providers.duckduckgo import DuckDuckGoAPIWrapper
from .providers.exa import ExaAPIWrapper
from .providers.tavily import TavilyAPIWrapper
from .search import ValidatedSearch, Search
from .pipeline import Project
from .db import ProjectStateDB
from langchain.schema import Document
from .generator import ContentGenerator

# Constants
DEFAULT_CACHE_DIR = Path("./cache")
DEFAULT_PROJECT_DIR = Path("./")
DEFAULT_MODEL_PATH = "/Volumes/JMD/models/phi4-4bit"

# Component configuration
SEARCH_PROVIDERS = ["duckduckgo", "exa", "brave", "tavily"]
ACADEMIC_DOMAINS = ["arxiv.org", "scholar.google", "researchgate", "academia.edu", ".edu", ".ac."]
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md"]
CONTENT_TYPES = ["All", "Article", "Documentation", "Code", "Other"]
SOURCE_TYPES = ["All", "url", "file"]
VALIDITY_FILTERS = ["All", "Valid Only", "Invalid Only"]
RESULT_FILTERS = ["Show Cached Only", "Hide Low Score (<0.5)", "Academic Sources Only"]

# Initialize components
def init_components(
    model_path: str = DEFAULT_MODEL_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    project_dir: Path = DEFAULT_PROJECT_DIR
) -> Tuple[PhiValidator, DocumentValidator, ValidatedSearch, Collector, Monitor]:
    """Initialize all required components."""
    
    # Initialize MLX model
    phi_model = PhiValidator(
        model_path=model_path,
        max_tokens=1000,
        temperature=0.1
    )
    
    # Initialize document validator
    validator = DocumentValidator(
        max_tokens=1000,
        temperature=0.1,
        cache_dir=cache_dir
    )
    
    # Initialize search providers
    providers = {
        "exa": ExaAPIWrapper(),
        "brave": BraveAPIWrapper(),
        "tavily": TavilyAPIWrapper(),
        "duckduckgo": DuckDuckGoAPIWrapper()
    }
    
    # Initialize search client
    search_client = ValidatedSearch(providers, min_score=0.3)
    
    # Initialize collector
    collector = Collector(
        cache_dir=cache_dir,
        rate_limit=1.0,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Initialize monitor
    monitor = Monitor(
        project_id="autosynth-web",
        project_dir=project_dir
    )
    
    return phi_model, validator, search_client, collector, monitor

# State management
class AppState:
    """Manages state for the Gradio interface."""
    
    def __init__(self):
        """Initialize app state."""
        self.project_id: Optional[str] = None
        self.topic: Optional[str] = None
        self.base_path: Optional[Path] = None
        self.project: Optional[Project] = None
        self.searcher: Optional[Search] = None
        self.collector: Optional[Collector] = None
        self.processor: Optional[Processor] = None
        self.generator: Optional[ContentGenerator] = None
        self.state_db: Optional[ProjectStateDB] = None
        self.logger = logging.getLogger("autosynth.gradio")

    def init_project(self, project_id: str, topic: str, base_path: str = "~/.autosynth"):
        """Initialize or load a project."""
        self.project_id = project_id
        self.topic = topic
        self.base_path = Path(base_path).expanduser()
        
        # Initialize components
        self.project = Project(project_id, topic, self.base_path)
        self.searcher = Search()
        self.collector = Collector()
        self.processor = Processor()
        self.generator = ContentGenerator()
        self.state_db = ProjectStateDB()

# Create global state
state = AppState()

# Register cleanup
atexit.register(lambda: asyncio.run(state.cleanup()))

# Search functionality
async def generate_search_queries(topic: str, num_queries: int = 5) -> List[str]:
    """Generate diverse search queries for a topic."""
    prompt = f"""Generate {num_queries} diverse search queries to gather comprehensive data about: {topic}
    
    The queries should:
    1. Cover different aspects of the topic
    2. Include technical and academic sources
    3. Target high-quality resources like papers, textbooks, and documentation
    
    Format: Return only the queries, one per line."""
    
    try:
        response = await asyncio.to_thread(state.phi_model, prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries[:num_queries]
    except Exception as e:
        return [f"Error generating queries: {str(e)}"]

async def search_query(
    query: str,
    providers: List[str],
    num_results: int,
    progress: gr.Progress = gr.Progress()
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Perform validated search across providers."""
    progress(0, desc="Validating query...")
    
    # Validate query
    validation_prompt = f"Is this search query well-formed and specific enough: '{query}'? Answer YES or NO with brief reason."
    try:
        validation = await asyncio.to_thread(state.phi_model, validation_prompt)
        if "NO" in validation.upper():
            return [], {
                "status": "error",
                "message": f"Invalid query: {validation}",
                "metrics": {}
            }
    except Exception as e:
        state.monitor.add_log(f"Query validation failed: {str(e)}", style="yellow")
    
    progress(0.2, desc="Searching providers...")
    
    # Update monitor
    state.monitor.update_stage_progress(
        stage="discover",
        completed=0,
        total=num_results,
        status="running"
    )
    
    # Perform search
    try:
        results = await state.search_client.search(
            query=query,
            selected_providers=providers,
            num_results=num_results
        )
        
        metrics = state.search_client.get_stats()
        state.search_results = results
        
        # Update monitor
        state.monitor.update_stage_metrics("discover", {
            "urls_found": len(results),
            "unique_domains": len(set(r["url"].split("/")[2] for r in results if "url" in r)),
            "filtered_urls": metrics["validation_count"] - len(results)
        })
        
        progress(1.0, desc="Search completed")
        
        return results, {
            "status": "success",
            "message": f"Found {len(results)} results",
            "metrics": metrics
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        state.monitor.add_log(error_msg, style="red")
        return [], {
            "status": "error",
            "message": error_msg,
            "metrics": {}
        }

# Visualization helpers
def create_provider_stats_plot(results: List[Dict]) -> go.Figure:
    """Create provider statistics visualization."""
    if not results:
        return None
        
    provider_data = {}
    for r in results:
        provider = r.get("provider", "unknown")
        if provider not in provider_data:
            provider_data[provider] = {
                "count": 0,
                "avg_score": 0,
                "cached": 0
            }
        
        data = provider_data[provider]
        data["count"] += 1
        data["avg_score"] += r.get("score", 0)
        if r.get("cached", False):
            data["cached"] += 1
    
    # Calculate averages
    for provider in provider_data.values():
        if provider["count"] > 0:
            provider["avg_score"] /= provider["count"]
    
    # Create plot
    fig = go.Figure()
    providers = list(provider_data.keys())
    
    # Add traces
    fig.add_trace(go.Bar(
        name="Total Results",
        x=providers,
        y=[d["count"] for d in provider_data.values()],
        marker_color="blue"
    ))
    
    fig.add_trace(go.Bar(
        name="Cached Results",
        x=providers,
        y=[d["cached"] for d in provider_data.values()],
        marker_color="green"
    ))
    
    fig.add_trace(go.Scatter(
        name="Avg Score",
        x=providers,
        y=[d["avg_score"] for d in provider_data.values()],
        yaxis="y2",
        line=dict(color="red", width=2)
    ))
    
    fig.update_layout(
        title="Provider Statistics",
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Average Score",
            overlaying="y",
            side="right"
        ),
        barmode="group"
    )
    
    return fig

def filter_results(results: List[Dict], filters: List[str]) -> List[Dict]:
    """Filter search results based on criteria."""
    filtered = results.copy()
    
    if "Show Cached Only" in filters:
        filtered = [r for r in filtered if r.get("cached", False)]
    
    if "Hide Low Score (<0.5)" in filters:
        filtered = [r for r in filtered if r.get("score", 0) >= 0.5]
    
    if "Academic Sources Only" in filters:
        filtered = [r for r in filtered if any(
            domain in r.get("url", "").lower()
            for domain in ACADEMIC_DOMAINS
        )]
    
    return filtered

# Interface builder functions
def build_search_tab() -> gr.Tab:
    """Build search tab interface."""
    with gr.Tab("Search") as tab:
        gr.Markdown("## Search Providers")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your query here",
                    lines=2
                )
                
                with gr.Row():
                    provider_selector = gr.CheckboxGroup(
                        label="Providers",
                        choices=SEARCH_PROVIDERS,
                        value=["duckduckgo"],
                        interactive=True
                    )
                    num_results_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
            
            with gr.Column(scale=2):
                search_metrics = gr.JSON(
                    label="Search Metrics",
                    value={"status": "ready", "metrics": state.search_client.get_stats()}
                )
        
        with gr.Row():
            search_btn = gr.Button("Search", variant="primary")
            clear_cache_btn = gr.Button("Clear Cache")
        
        with gr.Row():
            with gr.Column():
                results_table = gr.Dataframe(
                    headers=["title", "snippet", "url", "score", "provider", "cached"],
                    datatype=["str", "str", "str", "number", "str", "bool"],
                    interactive=False,
                    label="Search Results"
                )
            
            with gr.Column():
                result_filters = gr.CheckboxGroup(
                    label="Result Filters",
                    choices=RESULT_FILTERS,
                    value=[]
                )
                
                provider_stats = gr.Plot(label="Provider Statistics")
        
        # Connect components
        async def handle_search(query, providers, n):
            if not query.strip():
                return {
                    results_table: [],
                    search_metrics: {"status": "error", "message": "Empty query provided", "metrics": {}},
                    provider_stats: None
                }
            results, metrics = await search_query(query, providers, n)
            filtered_results = filter_results(results, result_filters.value)
            stats_plot = create_provider_stats_plot(results)
            return {
                results_table: filtered_results,
                search_metrics: metrics,
                provider_stats: stats_plot
            }
        
        search_btn.click(
            fn=handle_search,
            inputs=[query_box, provider_selector, num_results_slider],
            outputs=[results_table, search_metrics, provider_stats]
        )
        
        # Update results when filters change
        result_filters.change(
            fn=lambda results, filters: filter_results(results, filters),
            inputs=[results_table, result_filters],
            outputs=results_table
        )
        
        # Clear cache button
        async def clear_search_cache():
            state.search_client.vector_store.clear_all()
            return {
                "status": "success",
                "message": "Cache cleared",
                "metrics": state.search_client.get_stats()
            }
        
        async def clear_search_cache_wrapper():
            return await clear_search_cache()
        
        clear_cache_btn.click(
            fn=clear_search_cache_wrapper,
            inputs=[],
            outputs=[search_metrics]
        )
    
    return tab

# Additional visualization helpers
def create_validation_plots(validation_status: Dict[str, Any]) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Create visualization plots for validation results."""
    if not validation_status or "details" not in validation_status:
        return None, None, None
        
    # Type distribution pie chart
    type_data = validation_status["details"].get("by_type", {})
    type_fig = px.pie(
        values=list(type_data.values()),
        names=list(type_data.keys()),
        title="Content Type Distribution"
    )
    
    # Source distribution bar chart
    source_data = validation_status["details"].get("by_source", {})
    source_fig = px.bar(
        x=list(source_data.keys()),
        y=list(source_data.values()),
        title="Source Distribution"
    )
    
    # Validation times box plot
    times = validation_status["details"].get("validation_times", [])
    time_fig = px.box(
        y=times,
        title="Validation Time Distribution (seconds)"
    )
    
    return type_fig, source_fig, time_fig

def create_progress_plots() -> Tuple[go.Figure, go.Figure]:
    """Create progress visualization plots."""
    # Stage progress bar chart
    stages_data = []
    for stage, data in state.monitor.stage_progress.items():
        progress = (data["completed"] / data["total"] * 100) if data["total"] > 0 else 0
        stages_data.append({
            "stage": stage.capitalize(),
            "progress": progress,
            "status": data["status"]
        })
    
    progress_fig = px.bar(
        stages_data,
        x="stage",
        y="progress",
        color="status",
        title="Pipeline Progress"
    )
    
    # Metrics line chart for current stage
    current_stage = next(
        (stage for stage, data in state.monitor.stage_progress.items() 
         if data["status"] == "running"),
        "discover"
    )
    
    metrics = state.monitor.stage_metrics[current_stage]
    metrics_fig = go.Figure()
    
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics_fig.add_trace(
                go.Scatter(
                    x=[datetime.now()],
                    y=[value],
                    name=metric.replace("_", " ").title()
                )
            )
    
    metrics_fig.update_layout(title=f"{current_stage.capitalize()} Metrics")
    
    return progress_fig, metrics_fig

# Processing functionality
async def collect_documents(state: AppState, urls: str, progress=gr.Progress()):
    """Collect documents from provided URLs."""
    if not state.collector:
        return "Error: Project not initialized", {}
        
    try:
        # Parse URLs
        url_list = [url.strip() for url in urls.split("\n") if url.strip()]
        if not url_list:
            return "Error: No valid URLs provided", {}
            
        # Collect documents
        collected_docs = []
        for url in progress.tqdm(url_list, desc="Collecting documents"):
            try:
                docs = await state.collector.collect_batch(
                    url,
                    max_tokens=2000,
                    timeout=30
                )
                collected_docs.extend(docs)
            except Exception as e:
                state.logger.error(f"Error collecting from {url}: {str(e)}")
                continue
                
        # Store documents
        for doc in collected_docs:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source_type": "web",
                "tokens": len(doc.page_content.split())
            }
            state.state_db.add_collected_doc(state.project_id, doc_dict)
            
        # Get updated counts
        doc_count = state.state_db.get_doc_count(state.project_id, "collected")
        token_count = sum(len(doc.page_content.split()) for doc in collected_docs)
            
        return (
            f"Collected {len(collected_docs)} documents",
            {
                "documents_collected": doc_count,
                "tokens_collected": token_count
            }
        )
            
    except Exception as e:
        state.logger.error(f"Error in document collection: {str(e)}")
        return f"Error: {str(e)}", {}

async def process_documents(state: AppState, batch_size: int = 10, progress=gr.Progress()):
    """Process collected documents."""
    if not state.processor:
        return "Error: Project not initialized", {}
        
    try:
        # Get collected documents
        collected_docs = state.state_db.get_collected_docs(state.project_id)
        if not collected_docs:
            return "Error: No documents to process", {}
            
        processed_count = 0
        total_chunks = 0
        
        # Process in batches
        for i in progress.tqdm(
            range(0, len(collected_docs), batch_size),
            desc="Processing documents"
        ):
            batch = collected_docs[i:i + batch_size]
            
            for doc_dict in batch:
                try:
                    doc = Document(
                        page_content=doc_dict["content"],
                        metadata=doc_dict["metadata"]
                    )
                    
                    clean_doc = await state.processor.clean_content(
                        doc,
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    
                    if await state.processor.verify_quality(clean_doc, state.topic):
                        processed_dict = {
                            "content": clean_doc.page_content,
                            "metadata": clean_doc.metadata
                        }
                        state.state_db.add_processed_doc(state.project_id, processed_dict)
                        processed_count += 1
                        total_chunks += len(clean_doc.page_content.split())
                        
                except Exception as e:
                    state.logger.error(f"Error processing document: {str(e)}")
                    continue
                    
        return (
            f"Processed {processed_count} documents",
            {
                "documents_processed": processed_count,
                "chunks_created": total_chunks
            }
        )
            
    except Exception as e:
        state.logger.error(f"Error in document processing: {str(e)}")
        return f"Error: {str(e)}", {}

# Generation functionality
async def generate_content(doc_index: int, prompt: str, temperature: float) -> str:
    """Generate new content using Phi model."""
    if doc_index < 0 or doc_index >= len(state.validated_docs):
        return "Invalid document index."

    # Get the validated document
    doc_info = state.validated_docs[doc_index]
    
    # Update model temperature
    state.phi_model.temperature = temperature
    
    # Format prompt with context
    context = f"Original document: {doc_info.metadata.get('file_name', 'No name')}\n\n"
    full_prompt = context + prompt
    
    try:
        # Generate response
        generated_text = await asyncio.to_thread(state.phi_model, full_prompt)
        
        # Store generation
        state.generated_content.append({
            "original_doc_id": doc_index,
            "prompt": prompt,
            "temperature": temperature,
            "generated_text": generated_text
        })
        
        return generated_text
    except Exception as e:
        return f"Generation failed: {str(e)}"

# Monitoring functionality
def get_pipeline_logs() -> str:
    """Get recent logs from monitor."""
    logs = []
    for log in state.monitor.recent_logs[-10:]:  # Get last 10 logs
        time_str = log["time"].strftime("%H:%M:%S")
        logs.append(f"[{time_str}] {log['message']}")
    return "\n".join(logs)

def get_progress_metrics() -> Dict[str, Any]:
    """Get current progress metrics from monitor."""
    metrics = {
        "stages": state.monitor.stage_progress,
        "current_metrics": {},
        "last_update": datetime.now().strftime("%H:%M:%S")
    }
    
    # Find current/last active stage
    current_stage = next(
        (stage for stage, data in state.monitor.stage_progress.items() 
         if data["status"] == "running"),
        next(
            (stage for stage, data in reversed(state.monitor.stage_progress.items()) 
             if data["status"] == "completed"),
            "discover"
        )
    )
    
    metrics["current_metrics"] = state.monitor.stage_metrics[current_stage]
    return metrics

def build_ranking_tab() -> gr.Tab:
    """Build ranking tab interface."""
    with gr.Tab("Ranking") as tab:
        gr.Markdown("## Re-Rank Results")
        gr.Markdown("Select result rows from the Search tab to re-rank them with new weights.")
        
        with gr.Row():
            with gr.Column():
                selected_indices = gr.Textbox(
                    label="Indices (comma-separated)",
                    placeholder="e.g. 0,1,2"
                )
                semantic_slider = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                    label="Semantic Weight"
                )
                domain_slider = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    step=0.1,
                    value=0.3,
                    label="Domain Weight"
                )
            
            with gr.Column():
                ranked_table = gr.Dataframe(
                    headers=["title", "snippet", "url", "score"],
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                    label="Re-Ranked Results"
                )
        
        re_rank_btn = gr.Button("Re-Rank")
        
        def rerank_results(indices_str: str, semantic_weight: float, domain_weight: float) -> List[Dict]:
            """Re-rank selected search results."""
            if not indices_str.strip() or not state.search_results:
                return []
            
            # Parse indices
            try:
                indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]
            except ValueError:
                return []
            
            # Get selected results
            subset = [state.search_results[i] for i in indices if i < len(state.search_results)]
            
            # Apply weights
            for res in subset:
                res["score"] = (
                    res["score"] * semantic_weight +
                    (0.1 * domain_weight if any(d in res["url"] for d in ACADEMIC_DOMAINS) else 0)
                )
            
            # Sort by score
            subset.sort(key=lambda x: x["score"], reverse=True)
            return subset
        
        re_rank_btn.click(
            fn=rerank_results,
            inputs=[selected_indices, semantic_slider, domain_slider],
            outputs=ranked_table
        )
    
    return tab

def build_processing_tab() -> gr.Tab:
    """Build processing tab interface."""
    with gr.Tab("Processing") as tab:
        gr.Markdown("## Document Collection & Validation")
        
        # Batch processing controls
        with gr.Row():
            with gr.Column():
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Batch Size"
                )
                max_concurrent = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Max Concurrent Batches"
                )
            
            with gr.Column():
                processing_timeout = gr.Number(
                    value=30,
                    label="Processing Timeout (seconds)"
                )
                retry_attempts = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Retry Attempts"
                )
        
        # Collection components
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(
                    label="URLs (one per line)",
                    lines=5,
                    placeholder="Enter URLs to collect from..."
                )
                file_upload = gr.File(
                    label="Upload Documents",
                    file_types=SUPPORTED_FILE_TYPES,
                    multiple=True
                )
            
            with gr.Column():
                topic_input = gr.Textbox(
                    label="Topic (for validation)",
                    placeholder="Enter topic for content validation..."
                )
        
        # Add filtering controls
        with gr.Row():
            gr.Dropdown(
                choices=VALIDITY_FILTERS,
                value="All",
                label="Validity Filter",
                interactive=True
            )
            gr.Dropdown(
                choices=CONTENT_TYPES,
                value="All",
                label="Content Type",
                interactive=True
            )
            gr.Dropdown(
                choices=SOURCE_TYPES,
                value="All",
                label="Source Type",
                interactive=True
            )
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                label="Minimum Quality Score",
                interactive=True
            )

        # Add progress visualization
        with gr.Row():
            gr.Plot(
                label="Processing Progress",
                value=create_progress_plots()[0]
            )
            gr.Plot(
                label="Validation Statistics",
                value=create_progress_plots()[1]
            )
        
        # Results table
        results_table = gr.DataFrame(
            headers=[
                "doc_id", "file_name", "is_valid",
                "content_type", "validation_reason",
                "quality_score", "source_type"
            ],
            label="Validation Results"
        )
        
        # Action buttons
        with gr.Row():
            collect_btn = gr.Button("Collect Documents")
            validate_btn = gr.Button("Validate Documents")
        
        # Status display
        with gr.Row():
            with gr.Column():
                collection_status = gr.JSON(
                    label="Collection Status",
                    visible=True
                )
                validation_status = gr.JSON(
                    label="Validation Status",
                    visible=True
                )
            
            with gr.Column():
                stats_plot = gr.Plot(label="Validation Statistics")
        
        # Connect components
        collect_btn.click(
            fn=collect_documents,
            inputs=[
                file_upload,
                url_input,
                batch_size,
                max_concurrent,
                processing_timeout,
                retry_attempts
            ],
            outputs=[collection_status]
        )
        
        validate_btn.click(
            fn=process_documents,
            inputs=[
                topic_input,
                batch_size,
                max_concurrent,
                processing_timeout
            ],
            outputs=[
                validation_status,
                results_table,
                stats_plot
            ]
        )
    
    return tab

def build_generation_tab() -> gr.Tab:
    """Build generation tab interface."""
    with gr.Tab("Generation") as tab:
        gr.Markdown("## Content Generation")
        
        with gr.Row():
            with gr.Column():
                doc_index = gr.Number(
                    label="Validated Document Index",
                    value=0
                )
                prompt_box = gr.Textbox(
                    label="Generation Prompt",
                    lines=3,
                    placeholder="E.g. Summarize the key points..."
                )
                temp_slider = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    value=0.7,
                    label="Temperature"
                )
            
            with gr.Column():
                gen_output = gr.Textbox(
                    label="Generated Variation",
                    lines=10,
                    interactive=False
                )
        
        generate_btn = gr.Button("Generate")
        
        async def generate_content_wrapper(idx: int, prompt: str, temp: float) -> str:
            return await generate_content(idx, prompt, temp)
        
        generate_btn.click(
            fn=generate_content_wrapper,
            inputs=[doc_index, prompt_box, temp_slider],
            outputs=gen_output
        )
    
    return tab

def build_monitoring_tab() -> gr.Tab:
    """Build monitoring tab interface."""
    with gr.Tab("Monitoring") as tab:
        gr.Markdown("## Pipeline Monitor")
        
        with gr.Row():
            # Pipeline progress
            with gr.Column():
                progress_plot = gr.Plot(label="Pipeline Progress")
                metrics_plot = gr.Plot(label="Current Stage Metrics")
            
            # Stage details
            with gr.Column():
                stage_metrics = gr.JSON(
                    label="Stage Metrics",
                    value=get_progress_metrics()
                )
        
        with gr.Row():
            # Recent logs
            logs_box = gr.Textbox(
                label="Recent Logs",
                value=get_pipeline_logs(),
                lines=10,
                interactive=False
            )
        
        # Auto-refresh components
        def update_monitoring():
            """Update monitoring components."""
            metrics = get_progress_metrics()
            logs = get_pipeline_logs()
            progress_fig, metrics_fig = create_progress_plots()
            return {
                progress_plot: progress_fig,
                metrics_plot: metrics_fig,
                stage_metrics: metrics,
                logs_box: logs
            }
        
        gr.on(
            fn=update_monitoring,
            inputs=None,
            outputs=[progress_plot, metrics_plot, stage_metrics, logs_box],
            every=2  # Update every 2 seconds
        )
    
    return tab

def build_interface() -> gr.Blocks:
    """Build complete Gradio interface."""
    with gr.Blocks(title="AutoSynth") as demo:
        gr.Markdown("# AutoSynth")
        gr.Markdown("An AI-powered document synthesis and content generation pipeline.")
        
        # Create tabs container
        tabs = gr.Tabs(selected=0)  # Start with first tab selected
        
        # Add all tabs within the container
        with tabs:
            # Search tab for discovering and collecting documents
            with gr.TabItem("Search"):
                build_search_tab()
            
            # Ranking tab for re-ranking and filtering results
            with gr.TabItem("Ranking"):
                build_ranking_tab()
            
            # Processing tab for document validation and analysis
            with gr.TabItem("Processing"):
                build_processing_tab()
            
            # Generation tab for content synthesis
            with gr.TabItem("Generation"):
                build_generation_tab()
            
            # Monitoring tab for pipeline progress and metrics
            with gr.TabItem("Monitoring"):
                build_monitoring_tab()
        
        # Add footer with version and status
        with gr.Row():
            gr.Markdown("AutoSynth v1.0.0 | Status: ")
            status_indicator = gr.Markdown(
                value="Ready",
                elem_classes=["status-indicator"]
            )
        
        # Update status indicator based on state
        def update_status():
            if state.initialized:
                return "Running"
            return "Ready"
        
        demo.load(
            fn=update_status,
            inputs=None,
            outputs=status_indicator,
            every=5  # Update every 5 seconds
        )
        
        # Add custom CSS for styling
        demo.style("""
            .status-indicator {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                background-color: #e0e0e0;
                color: #333;
            }
            
            .status-indicator:contains("Running") {
                background-color: #4CAF50;
                color: white;
            }
            
            /* Add tab styling */
            .tabs {
                margin-top: 1rem;
                margin-bottom: 1rem;
            }
            
            .tab-nav {
                border-bottom: 2px solid #eee;
                padding-bottom: 0.5rem;
            }
            
            .tab-nav button {
                margin-right: 1rem;
                padding: 0.5rem 1rem;
                border: none;
                background: none;
                cursor: pointer;
            }
            
            .tab-nav button.selected {
                border-bottom: 2px solid #2196F3;
                color: #2196F3;
            }
            
            .tab-content {
                padding: 1rem 0;
            }
        """)
    
    return demo

def main():
    """Launch Gradio interface."""
    demo = build_interface()
    demo.queue()  # Enable queuing for async support
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()