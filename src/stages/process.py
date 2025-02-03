import logging
from langchain.schema import Document

class ProcessStage:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    async def run(self, project, target_chunks: int):
        """
        Run document processing stage, extracted from pipeline.py:_run_process.
        
        Args:
            project: The Project instance
            target_chunks: The target number of chunks to process
        """
        self.logger.info(f"Starting processing for project {project.project_id}")
        config = project.state.stage_configs["process"]

        try:
            processed_count = 0
            collected_docs = project.state.get_collected_docs(project.state_db)

            for doc_dict in collected_docs:
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
                        project.state_db.add_processed_doc(project.project_id, processed_dict)
                        processed_count += 1
                        project.update_metrics("chunks", processed_count, target_chunks)

                except Exception as e:
                    self.logger.error(f"Error processing document: {str(e)}")
                    continue

        finally:
            project.state.last_stage = "process"
            project.save_state()