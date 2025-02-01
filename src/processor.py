"""
Document validation module for AutoSynth pipeline
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field

from .model import MistralValidator

class ValidationResult(BaseModel):
    """Simple validation result"""
    is_valid: bool = Field(..., description="Whether document passed validation")
    reason: str = Field(..., description="Reason for validation result")
    content_type: str = Field(..., description="Detected content type")

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class DocumentValidator:
    """Quick document validation for the pipeline"""
    
    def __init__(
        self,
        max_tokens: int = 100,
        temperature: float = 0.1,
        cache_dir: Optional[Path] = None
    ):
        # Initialize components
        self.llm = MistralValidator(
            max_tokens=max_tokens,
            temperature=temperature
        )
        self.cache_dir = cache_dir
        
        # Setup validation chain
        self.validation_chain = self._setup_validation_chain()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_validation_chain(self):
        """Setup quick validation chain"""
        validation_prompt = ChatPromptTemplate(messages=[
            SystemMessage(content=(
                "You are a quick document validator. "
                "Check if the content is:"
                "\n1. In English"
                "\n2. Not gibberish"
                "\n3. Relevant to the topic"
                "\n\nProvide a simple yes/no with brief reason."
            )),
            HumanMessagePromptTemplate.from_template(
                "Topic: {topic}\nContent: {content}\n\n"
                "Is this content valid and relevant? Answer briefly."
            )
        ])
        
        return create_structured_output_chain(
            ValidationResult,
            self.llm,
            validation_prompt,
            verbose=True
        )
        
    def _check_structure(self, content: str) -> bool:
        """Basic structural validation"""
        if not content.strip():
            return False
            
        # Check if mostly English characters
        english_ratio = len(re.findall(r'[a-zA-Z\s]', content)) / len(content)
        if english_ratio < 0.5:
            return False
            
        # Check for code presence
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', ';']
        has_code = any(indicator in content for indicator in code_indicators)
        
        return True
        
    def _detect_content_type(self, content: str) -> str:
        """Detect content type based on structure"""
        if any(indicator in content for indicator in ['def ', 'class ', 'import ']):
            return "code"
        elif len(content.split('\n')) > 20:
            return "article"
        else:
            return "text"
            
    async def validate_document(
        self,
        doc: Document,
        topic: str
    ) -> Tuple[bool, Dict]:
        """
        Quick validation following pipeline requirements:
        1. Structural check (is it code? English? not gibberish?)
        2. Quick relevance check with small LLM
        """
        try:
            # 1. Basic structural validation
            if not self._check_structure(doc.page_content):
                return False, {
                    "source": doc.metadata.get("source", ""),
                    "reason": "Failed structural validation",
                    "validated_at": time.time()
                }
                
            # 2. Quick relevance check
            validation = await self.validation_chain.arun(
                topic=topic,
                content=doc.page_content[:1000]  # Only check first 1000 chars for speed
            )
            
            if not validation.is_valid:
                return False, {
                    "source": doc.metadata.get("source", ""),
                    "reason": validation.reason,
                    "validated_at": time.time()
                }
                
            # Document passed all checks
            metadata = {
                "source": doc.metadata.get("source", ""),
                "content_type": self._detect_content_type(doc.page_content),
                "validation_reason": validation.reason,
                "validated_at": time.time()
            }
            
            return True, metadata
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False, {
                "source": doc.metadata.get("source", ""),
                "reason": f"Validation error: {str(e)}",
                "validated_at": time.time()
            }
            
    async def validate_batch(
        self,
        docs: List[Document],
        topic: str,
        batch_size: int = 5
    ) -> Tuple[List[Document], List[Dict]]:
        """Validate documents in batches"""
        valid_docs = []
        validation_results = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.logger.info(f"Validating batch {i//batch_size + 1}")
            
            # Process batch concurrently
            tasks = [self.validate_document(doc, topic) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter valid documents
            for doc, (is_valid, metadata) in zip(batch, results):
                if is_valid:
                    doc.metadata.update(metadata)
                    valid_docs.append(doc)
                validation_results.append(metadata)
            
            self.logger.info(
                f"Batch complete: {len(valid_docs)}/{len(batch)} documents valid"
            )
            
        return valid_docs, validation_results 