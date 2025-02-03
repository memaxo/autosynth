import logging
import asyncio
from typing import Dict

from .stages.discover import DiscoverStage
from .stages.collect import CollectStage
from .stages.process import ProcessStage
from .stages.generate import GenerateStage

class PipelineManager:
    """
    Orchestrates running each pipeline stage.
    """
    def __init__(self, projects: Dict[str, "Project"]):
        """
        Args:
            projects: Dictionary of {project_id: Project} objects
        """
        self.logger = logging.getLogger("autosynth.pipeline")
        self.projects = projects

        # Instantiate each stage
        self.discover_stage = DiscoverStage(logger=self.logger)
        self.collect_stage = CollectStage(logger=self.logger)
        self.process_stage = ProcessStage(logger=self.logger)
        self.generate_stage = GenerateStage(logger=self.logger)

        # Map stage names to stage instances
        self.stage_map = {
            'discover': self.discover_stage,
            'collect': self.collect_stage,
            'process': self.process_stage,
            'generate': self.generate_stage
        }

    async def run_stage(self, project_id: str, stage: str, **targets):
        """
        Run a specific pipeline stage for a given project.
        
        Args:
            project_id: ID of the project to run a stage for
            stage: Name of the stage to run (discover, collect, process, generate)
            targets: Additional keyword args, e.g. target_urls=..., target_tokens=...
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]
        if stage not in self.stage_map:
            raise ValueError(f"Invalid stage: {stage}")

        # Example target metric name: 'target_discover_s'
        target_name = f"target_{stage}s"
        target_metric = targets.get(target_name)
        if target_metric is None:
            raise ValueError(f"No target specified for stage: {stage}")

        stage_runner = self.stage_map[stage]
        await stage_runner.run(project, target_metric)