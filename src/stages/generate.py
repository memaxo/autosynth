import logging
import asyncio
from langchain.schema import Document

class GenerateStage:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, project, target_variations: int):
        """
        Run content generation stage, extracted from pipeline.py:_run_generate.
        
        Args:
            project: The Project instance
            target_variations: Target number of variations to generate
        """
        self.logger.info(f"Starting generation for project {project.project_id}")
        config = project.state.stage_configs["generate"]

        try:
            processed_docs = project.state.get_processed_docs(project.state_db)
            generated_count = project.state_db.get_doc_count(project.project_id, "generated")

            for i in range(0, len(processed_docs), config["batch_size"]):
                if generated_count >= target_variations:
                    break

                batch = processed_docs[i:i + config["batch_size"]]
                batch_docs = [Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ) for doc in batch]

                for retry in range(config["max_retries"]):
                    try:
                        results = await project.generator.generate_batch(batch_docs)

                        for result in results:
                            if generated_count >= target_variations:
                                break

                            doc_dict = {
                                "content": result["generated_content"],
                                "original": result["original"],
                                "analysis": result["preservation_analysis"],
                                "metadata": result["metadata"]
                            }
                            project.state_db.add_generated_doc(project.project_id, doc_dict)
                            generated_count += 1

                        await project.update_metrics(
                            "variations",
                            generated_count,
                            target_variations
                        )
                        break  # Successful generation
                    except Exception as e:
                        self.logger.error(f"Error in batch generation (attempt {retry + 1}): {str(e)}")
                        if retry < config["max_retries"] - 1:
                            await asyncio.sleep(2 ** retry)
                        else:
                            self.logger.error(f"Failed to generate batch after {config['max_retries']} retries")
                            raise Exception(f"Batch generation failed after {config['max_retries']} retries")
                if generated_count >= target_variations:
                    break

        finally:
            project.state.last_stage = "generate"
            project.save_state()