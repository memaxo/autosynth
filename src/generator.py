from bespokelabs import curator
import logging
import os

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

class ContentGenerator(curator.LLM):
    """Handles content generation using DeepSeek R1 via Kluster.ai"""
    
    def __init__(self):
        super().__init__(
            model_name="deepseek-ai/DeepSeek-R1",
            backend="klusterai",
            batch=True,
            backend_params={
                "max_retries": 1,
                "completion_window": "1h"
            }
        )

    def prompt(self, input_doc):
        """Create prompt for content generation"""
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

    def parse(self, input_doc, response):
        """Parse the LLM response to extract reasoning and generated content"""
        import re

        # Extract reasoning and content
        reasoning_pattern = r"<think>(.*?)</think>"
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        content = re.sub(reasoning_pattern, "", response, flags=re.DOTALL).strip()

        return [{
            "original": input_doc['content'],
            "generated_content": content,
            "preservation_analysis": reasoning,
            "metadata": input_doc.get('metadata', {})
        }]

    async def generate_batch(self, documents):
        """Process a batch of documents"""
        inputs = [{"content": doc.page_content, "metadata": doc.metadata} 
                 for doc in documents]
        
        return self(inputs)  # Curator handles batching automatically