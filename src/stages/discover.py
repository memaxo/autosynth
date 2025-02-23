import logging
import json
import asyncio

class DiscoverStage:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, project, target_urls: int):
        """
        Run URL and GitHub repository discovery stage, extracted from pipeline.py:_run_discover.
        
        Args:
            project: The Project instance
            target_urls: The target number of URLs to discover
        """
        self.logger.info(f"Starting discovery for project {project.project_id}")
        config = project.state.stage_configs["discover"]
        def save_json(path, data):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

        try:
            # Web search discovery
            web_target = target_urls // 2
            while len(project.state.discovered_urls) < web_target:
                batch_size = min(
                    config["web_batch_size"],
                    web_target - len(project.state.discovered_urls)
                )

                try:
                    results = await project.searcher.search(
                        project.topic,
                        num_results=batch_size
                    )
                    new_urls = []
                    for r in results:
                        if "link" in r and r["link"] not in project.state.discovered_urls:
                            new_urls.append(r["link"])
                    project.state.discovered_urls.extend(new_urls)
                    await project.update_metrics("urls", len(project.state.discovered_urls), web_target)

                    # Save URLs asynchronously
                    urls_file = project.get_stage_data_path("discover") / "urls.json"
                    await asyncio.to_thread(save_json, urls_file, project.state.discovered_urls)

                except Exception as e:
                    self.logger.error(f"Error in web discovery: {str(e)}")
                    continue

            # GitHub repository discovery
            github_target = target_urls // 2
            while len(project.state.github_repos) < github_target:
                batch_size = min(
                    config["github_batch_size"],
                    github_target - len(project.state.github_repos)
                )

                try:
                    repos = await project.repo_collector.search_repositories(
                        project.topic,
                        limit=batch_size,
                        min_stars=100
                    )

                    project.state.github_repos.extend(repos)
                    await project.update_metrics("repos", len(project.state.github_repos), github_target)

                    # Save repos asynchronously
                    repos_file = project.get_stage_data_path("discover") / "github_repos.json"
                    await asyncio.to_thread(save_json, repos_file, project.state.github_repos)

                except Exception as e:
                    self.logger.error(f"Error in GitHub discovery: {str(e)}")
                    break

        finally:
            project.state.last_stage = "discover"
            project.save_state()