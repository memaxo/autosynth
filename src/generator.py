"""
Content generation module using DeepSeek-R1 model via Kluster.ai.

This module handles the generation of content variations while preserving
key information and technical accuracy of the original content.
"""

from bespokelabs import curator
import logging
import re
from typing import List, Dict, Any
from langchain.schema import Document

# Configure logging
logger = logging.getLogger(__name__)


class ContentGenerator(curator.LLM):
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

    def prompt(self, input_doc: Dict[str, Any]) -> str:
        """
        Create a prompt for content generation.
        
        Args:
            input_doc: Dictionary containing content and metadata
            
        Returns:
            Formatted prompt string for the model
        """
        return f"""
        Create a high-quality variation of the following content 
        while preserving the key information and technical accuracy.
        Make the content unique and well-structured.

        Original Content:
        {input_doc['content']}

        Generate the variation with this format:
        <think>Analysis and key points to preserve</think>
        [Generated variation here]
        """

    def parse(self, input_doc: Dict[str, Any], response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract reasoning and generated content.
        
        Args:
            input_doc: Original document dictionary
            response: Raw response from the model
            
        Returns:
            List containing parsed response with original content, generated content,
            preservation analysis, and metadata
        """
        try:
            # Extract reasoning section
            reasoning_match = re.search(
                r"<think>(.*?)</think>",
                response,
                re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Extract generated content
            content = re.sub(
                r"<think>.*?</think>",
                "",
                response,
                flags=re.DOTALL
            ).strip()
            
            if not content:
                logger.warning("Generated content is empty")
                return []

            return [{
                "original": input_doc['content'],
                "generated_content": content,
                "preservation_analysis": reasoning,
                "metadata": input_doc.get('metadata', {})
            }]
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            return []

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
            results = self(inputs)  # Curator handles batching
            
            if not results:
                logger.warning("No results generated for batch")
                
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise