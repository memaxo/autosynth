"""
Processor module for AutoSynth - handles content quality assessment and processing
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field

class ContentQuality(BaseModel):
    """Quality assessment output"""
    is_relevant: bool = Field(..., description="Whether content is relevant to topic")
    quality_score: float = Field(..., description="Quality score between 0 and 1")
    reasoning: str = Field(..., description="Reasoning for the assessment")
    suggested_improvements: List[str] = Field(..., description="Suggested improvements")

class ContentEnhancement(BaseModel):
    """Content enhancement output"""
    enhanced_content: str = Field(..., description="Enhanced version of the content")
    changes_made: List[str] = Field(..., description="List of changes made to improve the content")
    structure_score: float = Field(..., description="Score for content structure (0-1)")

class ProcessingError(Exception):
    """Raised when document processing fails"""
    pass

class Processor:
    """Handles document processing and quality assessment"""
    
    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        min_quality_score: float = 0.7,
        vectorstore_path: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        # Initialize components
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.min_quality_score = min_quality_score
        self.cache_dir = cache_dir
        
        # Setup vector store
        self.vectorstore = self._setup_vectorstore(vectorstore_path)
        
        # Initialize quality assessment chain
        self.quality_chain = self._setup_quality_chain()
        self.enhancement_chain = self._setup_enhancement_chain()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_vectorstore(self, path: Optional[str]) -> Chroma:
        """Setup vector store for similarity search"""
        return Chroma(
            collection_name="processed_docs",
            embedding_function=self.embeddings,
            persist_directory=path
        )
        
    def _setup_quality_chain(self):
        """Setup quality assessment chain"""
        quality_prompt = ChatPromptTemplate(messages=[
            SystemMessage(content=(
                "You are an expert content quality assessor. "
                "Analyze the content and determine if it is:"
                "\n1. Relevant to the topic"
                "\n2. High quality and informative"
                "\n3. Well-structured and clear"
                "\n\nProvide a detailed assessment with specific reasons and suggestions."
            )),
            HumanMessagePromptTemplate.from_template(
                "Topic: {topic}\nContent: {content}\n\n"
                "Assess the quality and provide structured feedback."
            )
        ])
        
        return create_structured_output_chain(
            ContentQuality,
            self.llm,
            quality_prompt,
            verbose=True
        )
        
    def _setup_enhancement_chain(self):
        """Setup content enhancement chain"""
        enhancement_prompt = ChatPromptTemplate(messages=[
            SystemMessage(content=(
                "You are an expert content enhancer. "
                "Improve the content while maintaining its core information by:"
                "\n1. Fixing formatting and structure"
                "\n2. Improving clarity and readability"
                "\n3. Standardizing style and terminology"
                "\n\nProvide the enhanced content and list specific improvements made."
            )),
            HumanMessagePromptTemplate.from_template(
                "Content: {content}\n\n"
                "Enhance this content while preserving its meaning."
            )
        ])
        
        return create_structured_output_chain(
            ContentEnhancement,
            self.llm,
            enhancement_prompt,
            verbose=True
        )
        
    async def _assess_quality(
        self,
        doc: Document,
        topic: str
    ) -> ContentQuality:
        """Assess document quality"""
        try:
            result = await self.quality_chain.arun(
                topic=topic,
                content=doc.page_content
            )
            return result
        except Exception as e:
            raise ProcessingError(f"Quality assessment failed: {str(e)}")
            
    async def _enhance_content(
        self,
        content: str
    ) -> ContentEnhancement:
        """Enhance document content"""
        try:
            result = await self.enhancement_chain.arun(content=content)
            return result
        except Exception as e:
            raise ProcessingError(f"Content enhancement failed: {str(e)}")
            
    async def _find_similar(
        self,
        embeddings: List[float],
        k: int = 3
    ) -> List[Document]:
        """Find similar documents"""
        try:
            return self.vectorstore.similarity_search_by_vector(embeddings, k=k)
        except Exception as e:
            self.logger.warning(f"Similarity search failed: {str(e)}")
            return []
            
    async def _compute_embeddings(
        self,
        content: str
    ) -> List[float]:
        """Compute content embeddings"""
        try:
            return await self.embeddings.aembed_query(content)
        except Exception as e:
            raise ProcessingError(f"Embedding computation failed: {str(e)}")
            
    def _update_vectorstore(
        self,
        doc: Document,
        embeddings: List[float]
    ):
        """Update vector store with new document"""
        try:
            self.vectorstore.add_embeddings(
                text_embeddings=[(doc.page_content, embeddings)],
                metadatas=[doc.metadata]
            )
        except Exception as e:
            self.logger.warning(f"Vector store update failed: {str(e)}")
            
    async def process_document(
        self,
        doc: Document,
        topic: str
    ) -> Optional[Document]:
        """Process single document"""
        try:
            # 1. Quality Assessment
            quality = await self._assess_quality(doc, topic)
            if not quality.is_relevant or quality.quality_score < self.min_quality_score:
                self.logger.info(
                    f"Document rejected: score={quality.quality_score}, "
                    f"reason={quality.reasoning}"
                )
                return None
                
            # 2. Content Enhancement
            enhanced = await self._enhance_content(doc.page_content)
            
            # 3. Semantic Analysis
            embeddings = await self._compute_embeddings(enhanced.enhanced_content)
            similar_docs = await self._find_similar(embeddings)
            
            # 4. Create Enhanced Document
            processed_doc = Document(
                page_content=enhanced.enhanced_content,
                metadata={
                    **doc.metadata,
                    "quality_score": quality.quality_score,
                    "quality_reasoning": quality.reasoning,
                    "suggested_improvements": quality.suggested_improvements,
                    "changes_made": enhanced.changes_made,
                    "structure_score": enhanced.structure_score,
                    "similar_docs": [d.metadata.get("source", "") for d in similar_docs],
                    "processing_time": time.time()
                }
            )
            
            # 5. Update Vector Store
            self._update_vectorstore(processed_doc, embeddings)
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return None
            
    async def process_batch(
        self,
        docs: List[Document],
        topic: str,
        batch_size: int = 5
    ) -> List[Document]:
        """Process documents in batches"""
        processed_docs = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}")
            
            # Process batch concurrently
            tasks = [self.process_document(doc, topic) for doc in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and None results
            valid_results = [
                doc for doc in results
                if doc is not None and not isinstance(doc, Exception)
            ]
            
            processed_docs.extend(valid_results)
            
            self.logger.info(
                f"Batch complete: {len(valid_results)}/{len(batch)} documents processed"
            )
            
        return processed_docs
        
    async def process(
        self,
        docs: List[Document],
        topic: str,
        batch_size: int = 5
    ) -> Tuple[List[Document], Dict]:
        """
        Process and validate documents
        
        Args:
            docs: List of documents to process
            topic: Topic for relevance checking
            batch_size: Size of batches for processing
            
        Returns:
            Tuple of (processed documents, processing stats)
        """
        start_time = time.time()
        
        try:
            processed_docs = await self.process_batch(docs, topic, batch_size)
            
            # Compute processing stats
            stats = {
                "total_docs": len(docs),
                "processed_docs": len(processed_docs),
                "avg_quality_score": sum(
                    float(d.metadata["quality_score"]) for d in processed_docs
                ) / len(processed_docs) if processed_docs else 0,
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(
                f"Processing complete: {stats['processed_docs']}/{stats['total_docs']} "
                f"documents processed in {stats['processing_time']:.2f}s"
            )
            
            return processed_docs, stats
            
        except Exception as e:
            self.logger.error(f"Processing pipeline failed: {str(e)}")
            raise 