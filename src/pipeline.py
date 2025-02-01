"""
AutoSynth pipeline orchestration module.

This module manages the end-to-end synthetic data generation pipeline, including:
- Project state management
- Stage execution (discover, collect, process, generate)
- Progress tracking and metrics
- Resource management
"""

import asyncio
import datetime
import json
import logging
import pickle
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain.callbacks import get_openai_callback
from langchain.schema import Document

from .collector import Collector
from .generator import ContentGenerator
from .github import GitHubCollector
from .processor import Processor
from .search import Search

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
class StageMetrics:
    """Track progress metrics for pipeline stages."""
    target_urls: int = 0
    current_urls: int = 0
    target_repos: int = 0
    current_repos: int = 0
    target_tokens: int = 0
    current_tokens: int = 0
    target_chunks: int = 0
    current_chunks: int = 0
    target_variations: int = 0
    current_variations: int = 0

@dataclass
class ProjectState:
    """Represents current state of a project."""
    discovered_urls: List[str] = field(default_factory=list)
    github_repos: List[Dict] = field(default_factory=list)
    collected_docs: List[Dict] = field(default_factory=list)
    processed_docs: List[Dict] = field(default_factory=list)
    generated_docs: List[Dict] = field(default_factory=list)
    last_stage: str = "init"
    last_updated: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metrics: StageMetrics = field(default_factory=StageMetrics)
    stage_configs: Dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())

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
        self.github_collector = GitHubCollector()
        self.collector = Collector()
        self.processor = Processor()
        self.generator = ContentGenerator()
        
        # Setup state and logging
        self.state = self._load_or_create_state()
        self.logger = logging.getLogger(f"autosynth.project.{project_id}")

    def _load_or_create_state(self) -> ProjectState:
        """Load existing state or create new one."""
        state_file = self.project_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                return ProjectState(**json.load(f))
        return ProjectState()

    def save_state(self):
        """Save current project state."""
        state_file = self.project_dir / "state.json"
        state_dict = asdict(self.state)
        state_dict["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def get_stage_data_path(self, stage: str) -> Path:
        """Get path for stage-specific data."""
        stage_dir = self.project_dir / stage
        stage_dir.mkdir(exist_ok=True)
        return stage_dir

    def update_metrics(self, stage: str, current: int, target: int):
        """Update progress metrics for a stage."""
        setattr(self.state.metrics, f"current_{stage}", current)
        setattr(self.state.metrics, f"target_{stage}", target)
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

    async def _run_discover(self, project: Project, target_urls: int):
        """Run URL and GitHub repository discovery stage."""
        self.logger.info(f"Starting discovery for project {project.project_id}")
        config = project.state.stage_configs["discover"]
        
        try:
            # Web search discovery
            web_target = target_urls // 2
            while len(project.state.discovered_urls) < web_target:
                batch_size = min(
                    config["web_batch_size"],
                    web_target - len(project.state.discovered_urls)
                )
                
                try:
                    results = await project.searcher.search_and_rank(
                        project.topic,
                        num_results=batch_size
                    )
                    
                    new_urls = [r["link"] for r in results if "link" in r]
                    project.state.discovered_urls.extend(new_urls)
                    project.update_metrics("urls", len(project.state.discovered_urls), web_target)
                    
                    # Save URLs
                    urls_file = project.get_stage_data_path("discover") / "urls.json"
                    with open(urls_file, 'w') as f:
                        json.dump(project.state.discovered_urls, f, indent=2)
                        
                except Exception as e:
                    self.logger.error(f"Error in web discovery: {str(e)}")
                    break

            # GitHub repository discovery
            github_target = target_urls // 2
            while len(project.state.github_repos) < github_target:
                batch_size = min(
                    config["github_batch_size"],
                    github_target - len(project.state.github_repos)
                )
                
                try:
                    repos = await project.github_collector.search_repositories(
                        project.topic,
                        limit=batch_size,
                        min_stars=100
                    )
                    
                    project.state.github_repos.extend(repos)
                    project.update_metrics("repos", len(project.state.github_repos), github_target)
                    
                    # Save repos
                    repos_file = project.get_stage_data_path("discover") / "github_repos.json"
                    with open(repos_file, 'w') as f:
                        json.dump(project.state.github_repos, f, indent=2)
                        
                except Exception as e:
                    self.logger.error(f"Error in GitHub discovery: {str(e)}")
                    break

        finally:
            project.state.last_stage = "discover"
            project.save_state()

    async def _run_collect(self, project: Project, target_tokens: int):
        """Run document collection stage from both web and GitHub sources."""
        self.logger.info(f"Starting collection for project {project.project_id}")
        config = project.state.stage_configs["collect"]
        
        try:
            current_tokens = 0
            web_target = target_tokens // 2
            
            # Web content collection
            for url in project.state.discovered_urls:
                if current_tokens >= web_target:
                    break
                    
                try:
                    docs = await project.collector.collect(
                        url, 
                        max_tokens=config["max_tokens_per_doc"],
                        timeout=config["timeout"]
                    )
                    
                    for doc in docs:
                        doc_dict = {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source_type": "web",
                            "tokens": len(doc.page_content.split())
                        }
                        
                        if current_tokens + doc_dict["tokens"] <= web_target:
                            project.state.collected_docs.append(doc_dict)
                            current_tokens += doc_dict["tokens"]
                            
                    project.update_metrics("tokens", current_tokens, target_tokens)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting from {url}: {str(e)}")
                    continue

            # GitHub content collection
            github_target = target_tokens // 2
            github_tokens = 0
            
            for repo in project.state.github_repos:
                if github_tokens >= github_target:
                    break
                    
                try:
                    repo_docs = await project.github_collector.collect_repository_content(
                        repo["full_name"],
                        max_files=config["github_max_files"]
                    )
                    
                    for doc in repo_docs:
                        doc_dict = {
                            "content": doc.page_content,
                            "metadata": {**doc.metadata, "repo": repo["full_name"]},
                            "source_type": "github",
                            "tokens": len(doc.page_content.split())
                        }
                        
                        if github_tokens + doc_dict["tokens"] <= github_target:
                            project.state.collected_docs.append(doc_dict)
                            github_tokens += doc_dict["tokens"]
                            current_tokens += doc_dict["tokens"]
                            
                    project.update_metrics("tokens", current_tokens, target_tokens)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting from GitHub repo {repo['full_name']}: {str(e)}")
                    continue

            # Save collected docs
            docs_file = project.get_stage_data_path("collect") / "collected_docs.pkl"
            with open(docs_file, 'wb') as f:
                pickle.dump(project.state.collected_docs, f)

        finally:
            project.state.last_stage = "collect"
            project.save_state()

    async def _run_process(self, project: Project, target_chunks: int):
        """Run document processing stage."""
        self.logger.info(f"Starting processing for project {project.project_id}")
        config = project.state.stage_configs["process"]
        
        try:
            processed_count = 0
            for doc_dict in project.state.collected_docs:
                if processed_count >= target_chunks:
                    break
                    
                try:
                    doc = Document(
                        page_content=doc_dict["content"],
                        metadata=doc_dict["metadata"]
                    )
                    
                    clean_doc = await project.processor.clean_content(
                        doc,
                        chunk_size=config["chunk_size"],
                        chunk_overlap=config["overlap"]
                    )
                    
                    if await project.processor.verify_quality(clean_doc, project.topic):
                        processed_dict = {
                            "content": clean_doc.page_content,
                            "metadata": clean_doc.metadata
                        }
                        project.state.processed_docs.append(processed_dict)
                        processed_count += 1
                        
                        project.update_metrics("chunks", processed_count, target_chunks)
                        
                        # Save processed docs periodically
                        if processed_count % 100 == 0:
                            docs_file = project.get_stage_data_path("process") / f"processed_{processed_count}.pkl"
                            with open(docs_file, 'wb') as f:
                                pickle.dump(project.state.processed_docs[-100:], f)
                        
                except Exception as e:
                    self.logger.error(f"Error processing document: {str(e)}")
                    continue

        finally:
            project.state.last_stage = "process"
            project.save_state()

    async def _run_generate(self, project: Project, target_variations: int):
        """Run content generation stage."""
        self.logger.info(f"Starting generation for project {project.project_id}")
        config = project.state.stage_configs["generate"]
        
        try:
            for i in range(0, len(project.state.processed_docs), config["batch_size"]):
                if len(project.state.generated_docs) >= target_variations:
                    break
                    
                batch = project.state.processed_docs[i:i + config["batch_size"]]
                batch_docs = [Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ) for doc in batch]
                
                for retry in range(config["max_retries"]):
                    try:
                        results = await project.generator.generate_batch(batch_docs)
                        
                        for result in results:
                            if len(project.state.generated_docs) >= target_variations:
                                break
                                
                            doc_dict = {
                                "content": result["generated_content"],
                                "original": result["original"],
                                "analysis": result["preservation_analysis"],
                                "metadata": result["metadata"]
                            }
                            project.state.generated_docs.append(doc_dict)
                            
                        project.update_metrics(
                            "variations",
                            len(project.state.generated_docs),
                            target_variations
                        )
                        
                        # Save batch
                        batch_file = project.get_stage_data_path("generate") / f"batch_{i}.pkl"
                        with open(batch_file, 'wb') as f:
                            pickle.dump(results, f)
                            
                        break  # Successful generation
                        
                    except Exception as e:
                        self.logger.error(f"Error in batch generation (attempt {retry + 1}): {str(e)}")
                        if retry < config["max_retries"] - 1:
                            await asyncio.sleep(2 ** retry)  # Exponential backoff
                        else:
                            self.logger.error(f"Failed to generate batch after {config['max_retries']} retries")

        finally:
            project.state.last_stage = "generate"
            project.save_state()

    async def run_stage(self, project_id: str, stage: str, **targets):
        """
        Run a specific pipeline stage for a project.
        
        Args:
            project_id: ID of project to run stage for
            stage: Pipeline stage to run
            **targets: Stage-specific target metrics
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
            
        project = self.projects[project_id]
        
        # Map stages to their implementation methods
        stage_methods = {
            'discover': self._run_discover,
            'collect': self._run_collect,
            'process': self._run_process,
            'generate': self._run_generate
        }
        
        if stage not in stage_methods:
            raise ValueError(f"Invalid stage: {stage}")
            
        # Get target metric for stage
        target_metric = targets.get(f'target_{stage}s', None)
        if target_metric is None:
            raise ValueError(f"No target specified for stage: {stage}")
            
        # Run stage
        try:
            await stage_methods[stage](project, target_metric)
        finally:
            self._save_projects()

def main():
    """CLI entry point for AutoSynth."""
    import argparse
    
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
        for pid, project in autosynth.projects.items():
            print(f"Project: {pid}")
            print(f"Topic: {project.topic}")
            print(f"Last Stage: {project.state.last_stage}")
            print(f"Last Updated: {project.state.last_updated}")
            print("Metrics:")
            for k, v in asdict(project.state.metrics).items():
                print(f"  {k}: {v}")
            print()
            
    elif args.new_project:
        project = autosynth.create_project(args.new_project)
        print(f"Created project {project.project_id}")
        
    elif args.project_id and args.stage:
        asyncio.run(autosynth.run_stage(
            args.project_id,
            args.stage,
            target_urls=args.target_urls,
            target_tokens=args.target_tokens,
            target_chunks=args.target_chunks,
            target_variations=args.target_variations
        ))

if __name__ == "__main__":
    main()
