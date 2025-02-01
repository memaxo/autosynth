"""
Terminal UI module for AutoSynth - handles progress monitoring and status display
"""

import time
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

class AutoSynthUI:
    """Terminal UI for monitoring AutoSynth pipeline progress"""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        
        # Initialize progress tracking
        self.stage_progress = {
            "discover": {"total": 0, "completed": 0, "status": "pending"},
            "collect": {"total": 0, "completed": 0, "status": "pending"},
            "process": {"total": 0, "completed": 0, "status": "pending"},
            "generate": {"total": 0, "completed": 0, "status": "pending"}
        }
        
        self.metrics = {}
        self.recent_logs: List[Dict] = []
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the UI layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", size=15),
            Layout(name="footer", size=6)
        )
        
        self.layout["main"].split_row(
            Layout(name="stages", ratio=2),
            Layout(name="metrics", ratio=1)
        )
        
    def _generate_header(self) -> Panel:
        """Generate header panel"""
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(
            Text("AutoSynth Pipeline Monitor", style="bold blue")
        )
        return Panel(grid, style="blue")
        
    def _generate_stages_panel(self) -> Panel:
        """Generate stages progress panel"""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Stage", justify="left")
        table.add_column("Progress", justify="center")
        table.add_column("Status", justify="right")
        
        for stage, data in self.stage_progress.items():
            # Calculate progress percentage
            progress = (data["completed"] / data["total"] * 100) if data["total"] > 0 else 0
            progress_str = f"{data['completed']}/{data['total']} ({progress:.1f}%)"
            
            # Status styling
            status_style = {
                "pending": "dim",
                "running": "yellow",
                "completed": "green",
                "error": "red"
            }.get(data["status"], "white")
            
            table.add_row(
                stage.capitalize(),
                progress_str,
                Text(data["status"], style=status_style)
            )
            
        return Panel(table, title="Pipeline Stages", border_style="blue")
        
    def _generate_metrics_panel(self) -> Panel:
        """Generate metrics panel"""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        
        for metric, value in self.metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            table.add_row(metric, formatted_value)
            
        return Panel(table, title="Current Metrics", border_style="blue")
        
    def _generate_logs_panel(self) -> Panel:
        """Generate recent logs panel"""
        table = Table(show_header=False, expand=True)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Message")
        
        for log in self.recent_logs[-5:]:  # Show last 5 logs
            table.add_row(
                log["time"].strftime("%H:%M:%S"),
                Text(log["message"], style=log.get("style", "white"))
            )
            
        return Panel(table, title="Recent Logs", border_style="blue")
        
    def update_stage_progress(
        self,
        stage: str,
        completed: int,
        total: int,
        status: str
    ):
        """Update progress for a pipeline stage"""
        if stage in self.stage_progress:
            self.stage_progress[stage].update({
                "completed": completed,
                "total": total,
                "status": status
            })
            
    def update_metrics(self, metrics: Dict):
        """Update current metrics"""
        self.metrics.update(metrics)
        
    def add_log(self, message: str, style: str = "white"):
        """Add a log message"""
        self.recent_logs.append({
            "time": datetime.now(),
            "message": message,
            "style": style
        })
        if len(self.recent_logs) > 100:  # Keep last 100 logs
            self.recent_logs.pop(0)
            
    def _render(self) -> Layout:
        """Render the complete UI"""
        self.layout["header"].update(self._generate_header())
        self.layout["stages"].update(self._generate_stages_panel())
        self.layout["metrics"].update(self._generate_metrics_panel())
        self.layout["footer"].update(self._generate_logs_panel())
        return self.layout
        
    def start(self):
        """Start the live UI"""
        with Live(
            self._render(),
            console=self.console,
            screen=True,
            refresh_per_second=4
        ) as live:
            while True:
                live.update(self._render())
                time.sleep(0.25)  # Update every 250ms
                
    def update(self):
        """Manual update of the UI - for use without live mode"""
        self.console.clear()
        self.console.print(self._render())
        
def create_progress_bar(description: str = "") -> Progress:
    """Create a rich progress bar with custom styling"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        expand=True,
        description=description
    ) 