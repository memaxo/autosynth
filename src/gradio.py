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

from .collector import Collector
from .model import PhiValidator
from .monitor import Monitor, create_progress_bar
from .processor import DocumentValidator
from .providers.brave import BraveAPIWrapper
from .providers.duckduckgo import DuckDuckGoAPIWrapper
from .providers.exa import ExaAPIWrapper
from .providers.tavily import TavilyAPIWrapper
from .search import ValidatedSearch

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
    """Manages application state and initialization."""
    
    def __init__(self):
        self.initialized = False
        self.cleanup_scheduled = False
        self.monitor_initialized = False
        self.last_update = datetime.now()
        
        # Data storage
        self.collected_docs: List[Dict] = []
        self.validated_docs: List[Dict] = []
        self.generated_content: List[Dict] = []
        self.search_results: List[Dict] = []
        
        # Initialize components
        (
            self.phi_model,
            self.validator,
            self.search_client,
            self.collector,
            self.monitor
        ) = init_components()
        
    async def initialize(self):
        """Initialize components if needed."""
        if not self.initialized:
            await self.collector.setup()
            self.initialized = True
            
    async def cleanup(self):
        """Cleanup components."""
        if self.initialized:
            await self.collector.cleanup()
            await self.validator.cleanup()
            self.initialized = False
            
    async def initialize_monitor(self):
        """Initialize monitor if needed."""
        if not self.monitor_initialized:
            asyncio.create_task(self.monitor.start())
            self.monitor_initialized = True

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
        def handle_search(query, providers, n):
            results, metrics = asyncio.run(search_query(query, providers, n))
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
            state.search_client.url_cache.clear()
            await state.search_client.milvus_store.clear()
            return {
                "status": "success",
                "message": "Cache cleared",
                "metrics": state.search_client.get_stats()
            }
        
        clear_cache_btn.click(
            fn=lambda: asyncio.run(clear_search_cache()),
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
async def collect_documents(
    file_objs: List[gr.File],
    urls: str = "",
    batch_size: int = 5,
    max_concurrent: int = 3,
    processing_timeout: float = 30,
    retry_attempts: int = 1,
    progress: gr.Progress = gr.Progress()
) -> Dict[str, Any]:
    """Collect documents from files and URLs."""
    await state.initialize()
    await state.initialize_monitor()
    
    # Update monitor stage
    state.monitor.update_stage_progress(
        stage="collect",
        completed=0,
        total=len(file_objs) + len(urls.split('\n')),
        status="running"
    )
    
    try:
        # Process URLs
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        collected_docs = await state.collector.collect_batch(
            urls=url_list,
            max_tokens=2000,
            timeout=processing_timeout,
            max_concurrency=max_concurrent
        )
        
        # Process files
        for file_obj in file_objs:
            try:
                docs = await state.collector.collect(
                    url=file_obj.name,
                    max_tokens=2000,
                    timeout=processing_timeout
                )
                collected_docs.extend(docs)
            except Exception as e:
                state.monitor.add_log(f"Failed to process {file_obj.name}: {str(e)}", style="red")
        
        # Update state
        state.collected_docs = collected_docs
        
        result = {
            "status": "success",
            "collected": len(collected_docs),
            "errors": []
        }
        
        # Update monitor
        state.monitor.update_stage_metrics("collect", {
            "documents_collected": len(collected_docs),
            "failed_downloads": len(result["errors"])
        })
        
        state.monitor.add_log(
            f"Collected {len(collected_docs)} documents",
            style="green"
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Collection failed: {str(e)}"
        state.monitor.add_log(error_msg, style="red")
        raise

async def validate_docs(
    topic: str = "general",
    batch_size: int = 5,
    max_concurrent: int = 3,
    processing_timeout: float = 30,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate collected documents."""
    await state.initialize_monitor()
    
    if not state.collected_docs:
        return {
            "status": "error",
            "message": "No documents to validate"
        }, []
    
    # Update monitor stage
    state.monitor.update_stage_progress(
        stage="process",
        completed=0,
        total=len(state.collected_docs),
        status="running"
    )
    
    try:
        # Process documents in batches
        validated_docs = []
        failed_docs = []
        
        for i in range(0, len(state.collected_docs), batch_size):
            batch = state.collected_docs[i:i + batch_size]
            
            try:
                results = await asyncio.gather(*[
                    state.validator.validate_document(
                        doc,
                        topic=topic,
                        timeout=processing_timeout
                    )
                    for doc in batch
                ], return_exceptions=True)
                
                for doc, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed_docs.append({
                            "doc_id": doc.metadata.get("doc_id"),
                            "error": str(result)
                        })
                    else:
                        validated_docs.append(result)
                        
            except Exception as e:
                state.monitor.add_log(f"Batch validation failed: {str(e)}", style="red")
        
        # Update state
        state.validated_docs = validated_docs
        
        # Prepare result data
        result = {
            "status": "success",
            "validated": len(validated_docs),
            "failed": len(failed_docs),
            "details": {
                "by_type": {},
                "by_source": {},
                "validation_times": []
            }
        }
        
        # Create table data
        table_data = [{
            "doc_id": doc.metadata.get("doc_id"),
            "file_name": doc.metadata.get("file_name", "Unknown"),
            "is_valid": doc.metadata.get("is_valid", False),
            "content_type": doc.metadata.get("content_type", "Unknown"),
            "validation_reason": doc.metadata.get("validation_reason", ""),
            "quality_score": doc.metadata.get("quality_score", 0.0),
            "source_type": doc.metadata.get("source_type", "unknown")
        } for doc in validated_docs]
        
        # Update monitor
        state.monitor.update_stage_metrics("process", {
            "documents_processed": len(validated_docs),
            "avg_quality_score": sum(d["quality_score"] for d in table_data) / len(table_data) if table_data else 0,
            "rejected_docs": len(failed_docs)
        })
        
        state.monitor.add_log(
            f"Validated {len(validated_docs)} documents ({len(failed_docs)} failed)",
            style="green"
        )
        
        return result, table_data
        
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        state.monitor.add_log(error_msg, style="red")
        raise

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
                validation_options = gr.CheckboxGroup(
                    choices=["Strict Mode", "Cache Results"],
                    label="Validation Options"
                )
        
        # Filtering controls
        with gr.Row():
            with gr.Column():
                validity_filter = gr.Radio(
                    choices=VALIDITY_FILTERS,
                    value="All",
                    label="Validity Filter"
                )
                content_type_filter = gr.Dropdown(
                    choices=CONTENT_TYPES,
                    value="All",
                    label="Content Type"
                )
            
            with gr.Column():
                source_type_filter = gr.Dropdown(
                    choices=SOURCE_TYPES,
                    value="All",
                    label="Source Type"
                )
                min_score_filter = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    label="Minimum Quality Score"
                )
        
        # Visualization components
        with gr.Row():
            with gr.Column():
                type_dist_plot = gr.Plot(label="Content Type Distribution")
                source_dist_plot = gr.Plot(label="Source Distribution")
            
            with gr.Column():
                time_dist_plot = gr.Plot(label="Validation Times")
                progress_chart = gr.Plot(label="Processing Progress")
        
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
            fn=validate_docs,
            inputs=[
                topic_input,
                batch_size,
                max_concurrent,
                processing_timeout
            ],
            outputs=[
                validation_status,
                results_table,
                type_dist_plot,
                source_dist_plot,
                time_dist_plot
            ]
        )
        
        # Add filtering
        def update_filtered_results(
            table_data: List[Dict],
            validity: str,
            content_type: str,
            source_type: str,
            min_score: float
        ) -> gr.DataFrame:
            """Filter validation results."""
            filters = {
                "validity": validity,
                "content_type": content_type,
                "source_type": source_type,
                "min_score": min_score
            }
            
            filtered = table_data.copy()
            
            if filters["validity"] != "All":
                is_valid = filters["validity"] == "Valid Only"
                filtered = [r for r in filtered if r["is_valid"] == is_valid]
            
            if filters["content_type"] != "All":
                filtered = [r for r in filtered if r["content_type"] == filters["content_type"]]
            
            if filters["source_type"] != "All":
                filtered = [r for r in filtered if r["source_type"] == filters["source_type"]]
            
            if filters["min_score"] > 0:
                filtered = [r for r in filtered if r["quality_score"] >= filters["min_score"]]
            
            return filtered
        
        # Connect filtering controls
        for filter_control in [
            validity_filter,
            content_type_filter,
            source_type_filter,
            min_score_filter
        ]:
            filter_control.change(
                fn=update_filtered_results,
                inputs=[
                    results_table,
                    validity_filter,
                    content_type_filter,
                    source_type_filter,
                    min_score_filter
                ],
                outputs=results_table
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
        
        def sync_generate_content(idx: int, prompt: str, temp: float) -> str:
            """Synchronous wrapper for content generation."""
            return asyncio.run(generate_content(idx, prompt, temp))
        
        generate_btn.click(
            fn=sync_generate_content,
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
        # Add all tabs
        search_tab = build_search_tab()
        ranking_tab = build_ranking_tab()
        processing_tab = build_processing_tab()
        generation_tab = build_generation_tab()
        monitoring_tab = build_monitoring_tab()
        
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