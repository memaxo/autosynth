import asyncio
import logging
import argparse
from pathlib import Path
import json
import datetime
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from langchain.schema import Document
from langchain.callbacks import get_openai_callback

from .searcher import Searcher
from .collector import Collector
from .processor import Processor
from .generator import ContentGenerator

@dataclass
class ProjectState:
    """Represents current state of a project"""
    discovered_urls: List[str]
    collected_docs: List[Dict]  # Serialized Document objects
    processed_docs: List[Dict]
    generated_docs: List[Dict]
    last_stage: str
    last_updated: str
    metrics: Dict[str, int]  # Progress tracking
    stage_configs: Dict[str, Dict]  # Stage-specific configurations

class Project:
    """Manages state and operations for a single synthetic data project"""
    
    def __init__(self, project_id: str, topic: str, base_path: Path):
        self.project_id = project_id
        self.topic = topic
        self.project_dir = base_path / project_id
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.searcher = Searcher()
        self.collector = Collector()
        self.processor = Processor()
        self.generator = ContentGenerator()
        
        # Load or create state
        self.state = self._load_or_create_state()
        self.logger = logging.getLogger(f"autosynth.project.{project_id}")

    def _load_or_create_state(self) -> ProjectState:
        """Load existing state or create new one"""
        state_file = self.project_dir / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                return ProjectState(**data)
        
        # Default stage configurations
        default_configs = {
            "discover": {"batch_size": 50, "min_quality_score": 0.7},
            "collect": {"max_tokens_per_doc": 2000, "timeout": 30},
            "process": {"chunk_size": 1000, "overlap": 200},
            "generate": {"batch_size": 5, "max_retries": 3}
        }
        
        return ProjectState(
            discovered_urls=[],
            collected_docs=[],
            processed_docs=[],
            generated_docs=[],
            last_stage="init",
            last_updated=datetime.datetime.now().isoformat(),
            metrics={
                "target_urls": 0,
                "current_urls": 0,
                "target_tokens": 0,
                "current_tokens": 0,
                "target_chunks": 0,
                "current_chunks": 0,
                "target_variations": 0,
                "current_variations": 0
            },
            stage_configs=default_configs
        )

    def save_state(self):
        """Save current project state"""
        state_file = self.project_dir / "state.json"
        state_dict = asdict(self.state)
        state_dict["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def get_stage_data_path(self, stage: str) -> Path:
        """Get path for stage-specific data"""
        stage_dir = self.project_dir / stage
        stage_dir.mkdir(exist_ok=True)
        return stage_dir

    def update_metrics(self, stage: str, current: int, target: int):
        """Update progress metrics for a stage"""
        self.state.metrics[f"current_{stage}"] = current
        self.state.metrics[f"target_{stage}"] = target
        self.save_state()

class AutoSynth:
    """Main AutoSynth tool managing multiple projects"""
    
    def __init__(self, base_path: str = "~/.autosynth"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.projects_file = self.base_path / "projects.json"
        self.projects: Dict[str, Project] = self._load_projects()
        
        # Setup logging
        self.logger = logging.getLogger("autosynth")
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_path / "autosynth.log")
            ]
        )

    def _load_projects(self) -> Dict[str, Project]:
        """Load existing projects"""
        if not self.projects_file.exists():
            return {}
            
        with open(self.projects_file) as f:
            project_data = json.load(f)
            
        return {
            pid: Project(pid, data['topic'], self.base_path)
            for pid, data in project_data.items()
        }

    def _save_projects(self):
        """Save projects metadata"""
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
        """Create a new project"""
        project_id = str(uuid.uuid4())[:8]
        project = Project(project_id, topic, self.base_path)
        
        self.projects[project_id] = project
        self._save_projects()
        
        self.logger.info(f"Created project {project_id} for topic: {topic}")
        return project

    async def _run_discover(self, project: Project, target_urls: int):
        """Run URL discovery stage"""
        self.logger.info(f"Starting discovery for project {project.project_id}")
        config = project.state.stage_configs["discover"]
        
        while len(project.state.discovered_urls) < target_urls:
            batch_size = min(config["batch_size"], 
                           target_urls - len(project.state.discovered_urls))
            
            try:
                results = await project.searcher.search_and_filter(
                    project.topic,
                    num_results=batch_size,
                    min_quality=config["min_quality_score"]
                )
                
                new_urls = [r["url"] for r in results]
                project.state.discovered_urls.extend(new_urls)
                project.update_metrics("urls", len(project.state.discovered_urls), target_urls)
                
                # Save URLs to stage-specific file
                urls_file = project.get_stage_data_path("discover") / "urls.json"
                with open(urls_file, 'w') as f:
                    json.dump(project.state.discovered_urls, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Error in discovery: {str(e)}")
                break

        project.state.last_stage = "discover"
        project.save_state()

    async def _run_collect(self, project: Project, target_tokens: int):
        """Run document collection stage"""
        self.logger.info(f"Starting collection for project {project.project_id}")
        config = project.state.stage_configs["collect"]
        
        current_tokens = 0
        for url in project.state.discovered_urls:
            if current_tokens >= target_tokens:
                break
                
            try:
                docs = await project.collector.collect(
                    url, 
                    max_tokens=config["max_tokens_per_doc"],
                    timeout=config["timeout"]
                )
                
                # Convert docs to serializable format
                for doc in docs:
                    doc_dict = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "tokens": len(doc.page_content.split())  # Simple approximation
                    }
                    
                    if current_tokens + doc_dict["tokens"] <= target_tokens:
                        project.state.collected_docs.append(doc_dict)
                        current_tokens += doc_dict["tokens"]
                        
                project.update_metrics("tokens", current_tokens, target_tokens)
                
                # Save batch to stage-specific directory
                docs_file = project.get_stage_data_path("collect") / f"batch_{len(project.state.collected_docs)}.pkl"
                with open(docs_file, 'wb') as f:
                    pickle.dump(docs, f)
                    
            except Exception as e:
                self.logger.error(f"Error collecting from {url}: {str(e)}")
                continue

        project.state.last_stage = "collect"
        project.save_state()

    async def _run_process(self, project: Project, target_chunks: int):
        """Run document processing stage"""
        self.logger.info(f"Starting processing for project {project.project_id}")
        config = project.state.stage_configs["process"]
        
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
                    
                    # Save processed docs
                    if processed_count % 100 == 0:
                        docs_file = project.get_stage_data_path("process") / f"processed_{processed_count}.pkl"
                        with open(docs_file, 'wb') as f:
                            pickle.dump(project.state.processed_docs[-100:], f)
                    
            except Exception as e:
                self.logger.error(f"Error processing document: {str(e)}")
                continue

        project.state.last_stage = "process"
        project.save_state()

    async def _run_generate(self, project: Project, target_variations: int):
        """Run content generation stage"""
        self.logger.info(f"Starting generation for project {project.project_id}")
        config = project.state.stage_configs["generate"]
        
        for i in range(0, len(project.state.processed_docs), config["batch_size"]):
            if len(project.state.generated_docs) >= target_variations:
                break
                
            batch = project.state.processed_docs[i:i + config["batch_size"]]
            batch_docs = [Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ) for doc in batch]
            
            retries = 0
            while retries < config["max_retries"]:
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
                    self.logger.error(f"Error in batch generation: {str(e)}")
                    retries += 1
                    await asyncio.sleep(2 ** retries)  # Exponential backoff
                    
            if retries == config["max_retries"]:
                self.logger.error(f"Failed to generate batch after {retries} retries")

        project.state.last_stage = "generate"
        project.save_state()

    async def run_stage(self, project_id: str, stage: str, **targets):
        """Run a specific stage for a project"""
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
        await stage_methods[stage](project, target_metric)
        
        self._save_projects()

def main():
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
            for k, v in project.state.metrics.items():
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
