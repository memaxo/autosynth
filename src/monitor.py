"""
Monitor module for AutoSynth pipeline progress and metrics tracking.

Features:
- Real-time progress tracking
- Stage-based metrics
- Performance monitoring
- Async-compatible UI updates
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

@dataclass
class StageProgress:
    """Track progress of a pipeline stage."""
    total: int = 0
    completed: int = 0
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@dataclass
class StageMetrics:
    """Stage-specific metrics tracking."""
    discover: Dict[str, Any] = field(default_factory=lambda: {
        "urls_found": 0,
        "unique_domains": 0,
        "filtered_urls": 0,
        "cache_hits": 0,
        "cache_misses": 0
    })
    collect: Dict[str, Any] = field(default_factory=lambda: {
        "documents_collected": 0,
        "total_tokens": 0,
        "failed_downloads": 0,
        "cached_docs": 0,
        "download_times": []
    })
    process: Dict[str, Any] = field(default_factory=lambda: {
        "documents_processed": 0,
        "avg_quality_score": 0.0,
        "rejected_docs": 0,
        "validation_times": [],
        "error_count": 0
    })
    generate: Dict[str, Any] = field(default_factory=lambda: {
        "variations_created": 0,
        "avg_novelty_score": 0.0,
        "failed_generations": 0,
        "generation_times": [],
        "token_usage": 0
    })

@dataclass
class LogEntry:
    """Structured log entry."""
    time: datetime
    message: str
    style: str = "white"
    level: str = "info"

class Monitor:
    """Monitor for tracking and displaying AutoSynth pipeline progress."""
    
    def __init__(self, project_id: str, project_dir: Path):
        """
        Initialize monitor with project context.
        
        Args:
            project_id: Unique identifier for the project
            project_dir: Project working directory
        """
        self.project_id = project_id
        self.project_dir = project_dir
        self.console = Console()
        
        # Initialize tracking components
        self.stage_progress = {
            stage: StageProgress() 
            for stage in ["discover", "collect", "process", "generate"]
        }
        self.metrics = StageMetrics()
        self.performance = {
            "cpu_usage": [],
            "memory_usage": [],
            "cache_stats": {"hits": 0, "misses": 0, "size": 0}
        }
        self.recent_logs: List[LogEntry] = []
        
        # Setup UI layout
        self._setup_layout()
        self._header_panel = self._render_header()
        self._stop = False
        
    def _setup_layout(self):
        """Initialize the UI layout structure."""
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=8)
        )
        self.layout["main"].split_row(
            Layout(name="stages", ratio=2),
            Layout(name="metrics", ratio=1)
        )
        self.layout["footer"].split_row(
            Layout(name="logs", ratio=2),
            Layout(name="performance", ratio=1)
        )

    def _create_table(self, title: str) -> Table:
        """Create a styled table with consistent formatting."""
        return Table(
            show_header=True,
            header_style=f"bold {title.lower()}",
            expand=True
        )

    def _format_time(self, start: Optional[datetime], end: Optional[datetime] = None) -> str:
        """Format time duration for display."""
        if not start:
            return ""
        end = end or datetime.now()
        return str(end - start).split('.')[0]

    def _format_value(self, value: Any) -> str:
        """Format metric values for display."""
        if isinstance(value, list):
            return f"{sum(value) / len(value):.2f} (avg)" if value else "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _render_header(self) -> Panel:
        """Render the header panel."""
        return Panel(
            Text(
                f"AutoSynth Pipeline Monitor - Project: {self.project_id}",
                style="bold blue",
                justify="center"
            ),
            style="blue"
        )

    def _render_stages(self) -> Panel:
        """Render the stages progress panel."""
        table = self._create_table("magenta")
        table.add_column("Stage")
        table.add_column("Progress", justify="center")
        table.add_column("Status", justify="right")
        table.add_column("Time", justify="right")
        
        for stage, progress in self.stage_progress.items():
            # Calculate progress percentage
            pct = (progress.completed / progress.total * 100) if progress.total > 0 else 0
            progress_str = f"{progress.completed}/{progress.total} ({pct:.1f}%)"
            
            # Status styling
            status_styles = {
                "pending": "dim",
                "running": "yellow",
                "completed": "green",
                "error": "red"
            }
            
            table.add_row(
                stage.capitalize(),
                progress_str,
                Text(progress.status, style=status_styles.get(progress.status, "white")),
                self._format_time(progress.start_time, progress.end_time)
            )
            
        return Panel(table, title="Pipeline Stages", border_style="blue")

    def _render_metrics(self) -> Panel:
        """Render the current stage metrics panel."""
        table = self._create_table("cyan")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        
        # Get current stage metrics
        current_stage = next(
            (stage for stage, prog in self.stage_progress.items() 
             if prog.status == "running"),
            next(
                (stage for stage, prog in reversed(self.stage_progress.items()) 
                 if prog.status == "completed"),
                "discover"
            )
        )
        
        stage_data = getattr(self.metrics, current_stage)
        for metric, value in stage_data.items():
            table.add_row(
                metric.replace("_", " ").title(),
                self._format_value(value)
            )
            
        return Panel(
            table,
            title=f"Current Stage Metrics ({current_stage.capitalize()})",
            border_style="blue"
        )

    def _render_performance(self) -> Panel:
        """Render the performance metrics panel."""
        table = self._create_table("green")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        
        # Add cache statistics
        cache_stats = self.performance["cache_stats"]
        total_cache_ops = cache_stats["hits"] + cache_stats["misses"]
        if total_cache_ops > 0:
            hit_ratio = cache_stats["hits"] / total_cache_ops * 100
            table.add_row("Cache Hit Ratio", f"{hit_ratio:.1f}%")
        
        # Add resource usage
        for metric in ["cpu_usage", "memory_usage"]:
            if self.performance[metric]:
                avg = sum(self.performance[metric]) / len(self.performance[metric])
                table.add_row(metric.replace("_", " ").title(), f"{avg:.1f}%")
            
        return Panel(table, title="Performance Metrics", border_style="blue")

    def _render_logs(self) -> Panel:
        """Render the recent logs panel."""
        table = Table(show_header=False, expand=True)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Message")
        
        for log in self.recent_logs[-8:]:
            table.add_row(
                log.time.strftime("%H:%M:%S"),
                Text(log.message, style=log.style)
            )
            
        return Panel(table, title="Recent Logs", border_style="blue")

    def update_stage_progress(self, stage: str, completed: int, total: int, status: str):
        """Update progress for a pipeline stage."""
        if stage in self.stage_progress:
            progress = self.stage_progress[stage]
            now = datetime.now()
            
            if status == "running" and progress.status != "running":
                progress.start_time = now
            elif status in ["completed", "error"] and progress.status == "running":
                progress.end_time = now
                
            progress.completed = completed
            progress.total = total
            progress.status = status

    def update_stage_metrics(self, stage: str, metrics: Dict[str, Any]):
        """Update metrics for a specific stage."""
        if hasattr(self.metrics, stage):
            stage_metrics = getattr(self.metrics, stage)
            for key, value in metrics.items():
                if key in stage_metrics:
                    if isinstance(stage_metrics[key], list):
                        stage_metrics[key].append(value)
                    else:
                        stage_metrics[key] = value

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        for key, value in metrics.items():
            if key in self.performance:
                if isinstance(self.performance[key], list):
                    self.performance[key].append(value)
                elif isinstance(self.performance[key], dict):
                    self.performance[key].update(value)
                else:
                    self.performance[key] = value

    def add_log(self, message: str, style: str = "white", level: str = "info"):
        """Add a log message."""
        log_entry = LogEntry(
            time=datetime.now(),
            message=message,
            style=style,
            level=level
        )
        self.recent_logs.append(log_entry)
        
        # Keep last 100 logs
        if len(self.recent_logs) > 100:
            self.recent_logs.pop(0)
        
        # Log to Python logger
        getattr(logger, level, logger.info)(message)

    def _render(self) -> Layout:
        """Render the complete UI."""
        self.layout["header"].update(self._header_panel)
        self.layout["stages"].update(self._render_stages())
        self.layout["metrics"].update(self._render_metrics())
        self.layout["logs"].update(self._render_logs())
        self.layout["performance"].update(self._render_performance())
        return self.layout

    async def start(self):
        """Start the monitor in live mode."""
        try:
            with Live(
                self._render(),
                console=self.console,
                screen=True,
                refresh_per_second=4
            ) as live:
                while not self._stop:
                    live.update(self._render())
                    await asyncio.sleep(0.25)
        except KeyboardInterrupt:
            self.console.print("\nMonitor stopped.")

    def update(self):
        """Manual update of the display."""
        self.console.clear()
        self.console.print(self._render())
    
    def stop(self):
        """Stop the monitor's live update loop."""
        self._stop = True

def create_progress_bar(description: str = "") -> Progress:
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
        description=description
    ) 