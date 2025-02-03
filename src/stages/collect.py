import logging
import asyncio

class CollectStage:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, project, target_tokens: int):
        """
        Run document collection stage from both web and GitHub sources, extracted from pipeline.py:_run_collect.
        
        Args:
            project: The Project instance
            target_tokens: The target number of tokens to collect
        """
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
                            project.state_db.add_collected_doc(project.project_id, doc_dict)
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
                    repo_docs = await project.repo_collector.collect_repository_content(
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
                            project.state_db.add_collected_doc(project.project_id, doc_dict)
                            github_tokens += doc_dict["tokens"]
                            current_tokens += doc_dict["tokens"]

                    project.update_metrics("tokens", current_tokens, target_tokens)

                except Exception as e:
                    self.logger.error(f"Error collecting from GitHub repo {repo['full_name']}: {str(e)}")
                    continue

        finally:
            project.state.last_stage = "collect"
            project.save_state()