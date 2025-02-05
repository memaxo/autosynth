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
            seen_urls = set()

        # Web content collection using concurrent tasks
            tasks = []
            for url in project.state.discovered_urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                tasks.append(self._collect_url(project, url, config, web_target))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for tokens in results:
                if isinstance(tokens, int):
                    current_tokens += tokens
                    if current_tokens >= web_target:
                        break
            await project.update_metrics("tokens", current_tokens, target_tokens)

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

    async def _collect_url(self, project, url, config, web_target):
        try:
            docs = await project.collector.collect(
                url,
                max_tokens=config["max_tokens_per_doc"],
                timeout=config["timeout"]
            )
            tokens_collected = 0
            for doc in docs:
                doc_tokens = len(doc.page_content.split())
                if tokens_collected + doc_tokens <= web_target:
                    doc_dict = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source_type": "web",
                        "tokens": doc_tokens
                    }
                    project.state_db.add_collected_doc(project.project_id, doc_dict)
                    tokens_collected += doc_tokens
            return tokens_collected
        except Exception as e:
            self.logger.error(f"Error collecting from {url}: {str(e)}")
            return 0