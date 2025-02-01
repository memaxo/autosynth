from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults, BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict
import aiohttp
import asyncio

class Searcher:
    """Handles automatic data discovery and initial filtering"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        self.search = DuckDuckGoSearchAPIWrapper()
        self.llm = ChatOpenAI(model_name=llm_model)
        self.embeddings = OpenAIEmbeddings()
        
    async def generate_search_queries(self, topic: str) -> List[str]:
        """Generate diverse search queries for the topic"""
        prompt = f"""
        Generate 5 different search queries to find high-quality educational content about {topic}.
        Focus on:
        - Academic sources
        - Technical documentation
        - Tutorial sites
        - Research papers
        - Expert blogs
        
        Format: Return only the queries, one per line.
        """
        
        response = await self.llm.apredict(prompt)
        return [q.strip() for q in response.split('\n') if q.strip()]
    
    async def _validate_url(self, url: str, topic: str) -> bool:
        """Quick validation of URL relevance"""
        prompt = f"""
        Given the URL: {url}
        Topic: {topic}
        
        Based on the URL alone, return 'True' only if this appears to be:
        1. A legitimate educational/technical resource
        2. Potentially relevant to the topic
        3. Not a social media or forum post
        4. Not an advertisement or product page
        
        Return only True or False:
        """
        response = await self.llm.apredict(prompt)
        return response.strip().lower() == 'true'
        
    async def search_and_filter(self, topic: str, num_results: int = 20) -> List[Dict]:
        """Perform search and initial filtering of results"""
        queries = await self.generate_search_queries(topic)
        all_results = []
        
        async def process_query(query: str):
            try:
                # Use DuckDuckGo API through LangChain
                results = self.search.results(query, num_results)
                
                # Initial filtering
                filtered_results = []
                for result in results:
                    if await self._validate_url(result["link"], topic):
                        filtered_results.append({
                            "url": result["link"],
                            "title": result["title"],
                            "snippet": result["snippet"]
                        })
                return filtered_results
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                return []

        # Process queries concurrently
        tasks = [process_query(query) for query in queries]
        results_lists = await asyncio.gather(*tasks)
        
        # Combine and deduplicate results
        seen_urls = set()
        for results in results_lists:
            for result in results:
                if result["url"] not in seen_urls:
                    all_results.append(result)
                    seen_urls.add(result["url"])
        
        return all_results
    
    async def rank_results(self, results: List[Dict], topic: str) -> List[Dict]:
        """Rank results by relevance using embeddings"""
        if not results:
            return []
            
        # Create topic embedding
        topic_emb = await self.embeddings.aembed_query(topic)
        
        # Create embeddings for each result
        for result in results:
            text = f"{result['title']} {result['snippet']}"
            result_emb = await self.embeddings.aembed_query(text)
            
            # Calculate similarity
            similarity = sum(a * b for a, b in zip(topic_emb, result_emb))
            result["relevance_score"] = similarity
        
        # Sort by relevance
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)