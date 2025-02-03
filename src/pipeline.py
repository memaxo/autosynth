"""
AutoSynth pipeline orchestration module (refactored).

Now uses PipelineManager plus new stage classes in src/stages/.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain.schema import Document

from .collector import Collector
from .generator import ContentGenerator
from .github import RepoCollector
from .processor import Processor
from .search import Search
from .db import ProjectStateDB

# Constants
DEFAULT_CONFIG = {
    "discover": {
        "web_batch_size": 50,
        "github_batch_size": 20,
        "min_quality_score": 0.7
    },
    "collect": {
        "max_tokens_per_doc": 2000,
        "timeout": 30,
        "github_max_files": 10
    },
    "process": {
        "chunk_size": 1000,
        "overlap": 200
    },
    "generate": {
        "batch_size": 5,
        "max_retries": 3
    }
}

@dataclass
class ProjectMetrics:
    """Tracks progress metrics for each stage."""
    current_urls: int = 0
    target_urls: int = 0
    current_repos: int = 0
    target_repos: int = 0
    current_tokens: int = 0
    target_tokens: int = 0
    current_chunks: int = 0
    target_chunks: int = 0
    current_variations: int = 0
    target_variations: int = 0

@dataclass
class ProjectState:
    """Represents the current state of a project."""
    last_stage: Optional[str] = None
    last_updated: Optional[str] = None
    stage_configs: Dict = field(default_factory=lambda: DEFAULT_CONFIG)
    discovered_urls: List[str] = field(default_factory=list)
    github_repos: List[Dict] = field(default_factory=list)
    metrics: ProjectMetrics = field(default_factory=ProjectMetrics)

    def get_collected_docs(self, db: ProjectStateDB) -> List[Dict]:
        """Get collected documents from database."""
        return db.get_collected_docs(self.project_id)

    def get_processed_docs(self, db: ProjectStateDB) -> List[Dict]:
        """Get processed documents from database."""
        return db.get_processed_docs(self.project_id)

    def get_generated_docs(self, db: ProjectStateDB) -> List[Dict]:
        """Get generated documents from database."""
        return db.get_generated_docs(self.project_id)

class Project:
    """Manages state and operations for a single synthetic data project."""
    
    def __init__(self, project_id: str, topic: str, base_path: Path):
        """
        Initialize project with components and state.
        
        Args:
            project_id: Unique identifier for the project
            topic: Project topic/domain
            base_path: Base directory for project files
        """
        self.project_id = project_id
        self.topic = topic
        self.project_dir = base_path / project_id
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.searcher = Search()
        self.repo_collector = RepoCollector()
        self.collector = Collector()
        self.processor = Processor()
        self.generator = ContentGenerator()

        # Initialize state management
        self.state_db = ProjectStateDB()
        self.state = self._load_or_create_state()
        self.logger = logging.getLogger(f"autosynth.project.{project_id}")

    def _load_or_create_state(self) -> ProjectState:
        """Load existing state or create new one."""
        project_data = self.state_db.get_project(self.project_id)
        if project_data:
            state = ProjectState()
            state.last_stage = project_data["last_stage"]
            state.last_updated = project_data["last_updated"]
            state.stage_configs = project_data["config"]

            # Load state data
            state.discovered_urls = self.state_db.get_urls(self.project_id)
            state.github_repos = self.state_db.get_github_repos(self.project_id)

            # Load metrics
            metrics = self.state_db.get_metrics(self.project_id)
            for name, values in metrics.items():
                setattr(state.metrics, f"current_{name}", values["current"])
                setattr(state.metrics, f"target_{name}", values["target"])

            return state

        # Create new state
        self.state_db.create_project(
            self.project_id,
            self.topic,
            DEFAULT_CONFIG
        )
        return ProjectState()

    def save_state(self):
        """Save current project state."""
        self.state_db.update_project_state(
            self.project_id,
            self.state.last_stage
        )

        # Save URLs and repos
        if self.state.discovered_urls:
            self.state_db.add_urls(self.project_id, self.state.discovered_urls)
        if self.state.github_repos:
            self.state_db.add_github_repos(self.project_id, self.state.github_repos)

        # Save metrics
        metrics = asdict(self.state.metrics)
        for name in ["urls", "repos", "tokens", "chunks", "variations"]:
            current = metrics.get(f"current_{name}", 0)
            target = metrics.get(f"target_{name}", 0)
            if current > 0 or target > 0:
                self.state_db.update_metrics(self.project_id, name, current, target)

    def get_stage_data_path(self, stage: str) -> Path:
        """Get path for stage-specific data."""
        stage_dir = self.project_dir / stage
        stage_dir.mkdir(exist_ok=True)
        return stage_dir

    def update_metrics(self, stage: str, current: int, target: int):
        """Update progress metrics for a stage."""
        setattr(self.state.metrics, f"current_{stage}", current)
        setattr(self.state.metrics, f"target_{stage}", target)
        self.state_db.update_metrics(self.project_id, stage, current, target)
        self.save_state()

class AutoSynth:
    """Main AutoSynth tool managing multiple projects."""
    
    def __init__(self, base_path: str = "~/.autosynth"):
        """
        Initialize AutoSynth with base configuration.
        
        Args:
            base_path: Base directory for all projects
        """
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.projects_file = self.base_path / "projects.json"
        self.projects: Dict[str, Project] = self._load_projects()

        # Setup logging
        self.logger = logging.getLogger("autosynth")
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging handlers."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_path / "autosynth.log")
            ]
        )

    def _load_projects(self) -> Dict[str, Project]:
        """Load existing projects from disk."""
        if not self.projects_file.exists():
            return {}

        with open(self.projects_file) as f:
            project_data = json.load(f)

        return {
            pid: Project(pid, data['topic'], self.base_path)
            for pid, data in project_data.items()
        }

    def _save_projects(self):
        """Save projects metadata to disk."""
        data = {
            pid: {
                'topic': project.topic,
                'last_updated': project.state.last_updated
            }
            for pid, project in self.projects.items()
        }

        with open(self.projects_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_project(self, topic: str) -> Project:
        """Create a new project for given topic."""
        project_id = str(uuid.uuid4())[:8]
        project = Project(project_id, topic, self.base_path)

        self.projects[project_id] = project
        self._save_projects()

        self.logger.info(f"Created project {project_id} for topic: {topic}")
        return project

    def list_projects(self):
        """List all projects."""
        for pid, project in self.projects.items():
            print(f"Project: {pid}")
            print(f"Topic: {project.topic}")
            print(f"Last Stage: {project.state.last_stage}")
            print(f"Last Updated: {project.state.last_updated}")
            print("Metrics:")
            for k, v in asdict(project.state.metrics).items():
                print(f"  {k}: {v}")
            print()

def main():
    """CLI entry point for AutoSynth."""
    import argparse
    from .pipeline_manager import PipelineManager

    parser = argparse.ArgumentParser(description='AutoSynth - Synthetic Dataset Generator')

    # Project management
    parser.add_argument('--new-project', type=str, 
                       help='Create new project with given topic')
    parser.add_argument('--list-projects', action='store_true',
                       help='List all projects')
    parser.add_argument('--project-id', type=str,
                       help='Project ID to work with')

    # Stage control
    parser.add_argument('--stage', type=str,
                       choices=['discover', 'collect', 'process', 'generate'],
                       help='Pipeline stage to run')

    # Metric targets
    parser.add_argument('--target-urls', type=int, default=100,
                       help='Target number of URLs to discover')
    parser.add_argument('--target-tokens', type=int, default=100000,
                       help='Target number of tokens to collect')
    parser.add_argument('--target-chunks', type=int, default=1000,
                       help='Target number of chunks to process')
    parser.add_argument('--target-variations', type=int, default=500,
                       help='Target number of variations to generate')

    args = parser.parse_args()

    # Initialize AutoSynth
    autosynth = AutoSynth()

    if args.list_projects:
        autosynth.list_projects()

    elif args.new_project:
        project = autosynth.create_project(args.new_project)
        print(f"Created project {project.project_id}")

    elif args.project_id and args.stage:
        # Use the new PipelineManager to run the specified stage
        manager = PipelineManager(autosynth.projects)
        asyncio.run(manager.run_stage(
            args.project_id,
            args.stage,
            target_urls=args.target_urls,
            target_tokens=args.target_tokens,
            target_chunks=args.target_chunks,
            target_variations=args.target_variations
        ))

if __name__ == "__main__":
    main()