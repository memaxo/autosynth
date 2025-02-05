"""
Content generation module using DeepSeek-R1 model via Kluster.ai.

This module handles the generation of content variations while preserving
key information and technical accuracy of the original content.
"""

from bespokelabs import curator
import logging
import re
import json
from typing import List, Dict, Any
from langchain.schema import Document
from .utils.prompts import GENERATION_PROMPT

# Configure logging
logger = logging.getLogger(__name__)


class ContentGenerator(curator.LLM):
    SYSTEM_PROMPT: str = "You are a content generator. Generate high-quality variations with technical accuracy."
    """
    Content generation using DeepSeek-R1 model via Kluster.ai.
    
    Handles batch processing of documents to create high-quality content variations
    while maintaining technical accuracy and key information.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        max_retries: int = 1,
        completion_window: str = "1h"
    ):
        """
        Initialize the content generator.
        
        Args:
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts for failed generations
            completion_window: Time window for batch completion
        """
        super().__init__(
            model_name=model_name,
            backend="klusterai",
            batch=True,
            backend_params={
                "max_retries": max_retries,
                "completion_window": completion_window
            }
        )
        logger.info(f"Initialized ContentGenerator with model: {model_name}")

    def prompt(self, input_doc: Dict[str, Any]) -> list:
        """
        Create a message-based prompt for content generation.
        
        Args:
            input_doc: Dictionary containing content and metadata. Must include a 'content' key.
            
        Returns:
            A list of messages formatted for the LLM.
        """
        if "content" not in input_doc or not input_doc["content"]:
            raise ValueError("Input document must contain a non-empty 'content' key.")
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"{GENERATION_PROMPT}\n{input_doc['content']}"}
        ]

    def parse(self, input_doc: Dict[str, Any], response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response (expected in JSON format) to extract generated content.
        
        Args:
            input_doc: Original document dictionary
            response: Raw JSON response from the model
            
        Returns:
            List containing a dictionary with original content, generated content,
            preservation analysis, verification, and metadata.
        """
        try:
            import json
            response_data = json.loads(response)
            return [{
                "original": input_doc["content"],
                "generated_content": response_data.get("variation", ""),
                "preservation_analysis": response_data.get("analysis", ""),
                "verification": response_data.get("verification", ""),
                "metadata": input_doc.get("metadata", {})
            }]
        except Exception as e:
            logger.error(f"Error parsing generation response: {e}")
            raise ValueError(f"Error parsing generation response: {e}")

    async def generate_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents to generate content variations.
        
        Args:
            documents: List of Document objects to process
            
        Returns:
            List of dictionaries containing generated content and metadata
            
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("No documents provided for generation")
            
        try:
            inputs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
            logger.info(f"Processing batch of {len(documents)} documents")
            results = await asyncio.to_thread(self, inputs)  # Execute generation in a separate thread
            
            if not results:
                logger.warning("No results generated for batch")
            else:
                for idx, item in enumerate(results):
                    if not isinstance(item, dict) or "generated_content" not in item:
                        logger.warning(f"Result at index {idx} has an unexpected format: {item}")
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise