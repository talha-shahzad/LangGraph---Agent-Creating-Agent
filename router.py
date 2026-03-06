import hashlib
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from registry import ToolRegistry

class DeterministicRouter:
    """
    Handles deterministic query routing using semantic embeddings and result caching.
    """
    
    def __init__(self, registry: ToolRegistry, embedding_model: GoogleGenerativeAIEmbeddings, threshold: float = 0.7):
        self.registry = registry
        self.embeddings = embedding_model
        self.threshold = threshold

    def _generate_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generates a stable hash key for a tool call."""
        # Sort params to ensure deterministic hashing
        params_str = json.dumps(params, sort_keys=True)
        raw_key = f"{tool_name}:{params_str}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    async def get_query_embedding(self, query: str) -> List[float]:
        """Fetches embedding for the user query."""
        return await self.embeddings.aembed_query(query)

    async def route_query(self, query: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        Routes the query to an existing tool, cache, or generator.
        Returns: (route, tool_data, cached_result)
        Routes: 'cache', 'execute_existing', 'generate_new'
        """
        # 1. Get query embedding
        query_emb = await self.get_query_embedding(query)
        
        # 2. Search for matching tools
        matches = self.registry.search_tools(query_emb, float(self.threshold))
        logging.info(f"Router found {len(matches)} matches for threshold {self.threshold}")
        
        if matches:
            # Select the best match
            best_match = matches[0]
            logging.info(f"Router selecting best match: {best_match['name']} (Sim: {best_match.get('similarity', 'N/A')})")
            
            # Note: Parameter extraction still needs the query context.
            # For the deterministic router, we find the tool first.
            return "execute_existing", best_match, None
        
        return "generate_new", None, None

    def check_cache(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """Checks if a similar tool execution exists in cache."""
        cache_key = self._generate_cache_key(tool_name, params)
        return self.registry.get_cached_result(cache_key)

    def cache_result(self, tool_name: str, params: Dict[str, Any], result: str):
        """Stores execution result in cache."""
        cache_key = self._generate_cache_key(tool_name, params)
        self.registry.set_cache_result(cache_key, result)
