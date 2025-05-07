import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import yaml
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

class GraphRAG:
    def __init__(self, config_path: str = "config/neo4j_config.yaml"):
        self.config = self._load_config(config_path)
        self.driver = self._init_driver()
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.vector_store = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_driver(self) -> GraphDatabase.driver:
        """Initialize Neo4j driver."""
        uri = os.getenv("NEO4J_URI", self.config["neo4j"]["uri"])
        user = os.getenv("NEO4J_USER", self.config["neo4j"]["user"])
        password = os.getenv("NEO4J_PASSWORD", self.config["neo4j"]["password"])
        return GraphDatabase.driver(uri, auth=(user, password))

    def get_graph_context(self, query: str, user_id: Optional[str] = None, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the knowledge graph with community awareness."""
        # First, find anchor nodes using semantic search
        anchor_nodes = self._find_anchor_nodes(query)
        
        # Get user's community if user_id is provided
        user_community = self._get_user_community(user_id) if user_id else None
        
        # Then, traverse the graph to get context
        context = []
        with self.driver.session() as session:
            for anchor in anchor_nodes:
                # Get product context
                product_context = self._get_product_context(session, anchor["asin"], max_hops)
                context.extend(product_context)
                
                # Get community context if available
                if user_community:
                    community_context = self._get_community_context(
                        session, 
                        anchor["asin"], 
                        user_community,
                        max_hops
                    )
                    context.extend(community_context)
        
        return self._deduplicate_context(context)

    def get_semantic_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve context using pure semantic search."""
        query_embedding = self.embeddings.embed_query(query)
        
        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('product_embeddings', $top_k, $embedding)
                YIELD node, score
                RETURN node.asin as asin,
                       node.title as title,
                       node.description as description,
                       node.price as price,
                       score
                """,
                embedding=query_embedding,
                top_k=top_k
            )
            
            return [dict(record) for record in result]

    def answer_query(self, query: str, user_id: Optional[str] = None, method: str = "graph") -> Tuple[str, List[Dict[str, Any]]]:
        """Answer a query using either GraphRAG or semantic retrieval.
        
        Args:
            query: The user's query
            user_id: Optional user ID for community-aware recommendations
            method: Either "graph" or "semantic" to specify retrieval method
            
        Returns:
            Tuple of (answer, retrieved_context)
        """
        if method == "graph":
            context = self.get_graph_context(query, user_id)
        else:
            context = self.get_semantic_context(query)
        
        # Format context for the LLM
        formatted_context = self._format_context(context)
        
        # Generate answer using the LLM
        prompt = f"""
        Context:
        {formatted_context}
        
        User Query: {query}
        
        Based on the context above, please provide a detailed answer to the user's query.
        
        Answer:
        """
        
        answer = self.llm(prompt)
        return answer, context

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for the LLM."""
        formatted = []
        for item in context:
            if isinstance(item, dict) and 'title' in item:  # Semantic search result
                formatted.append(
                    f"Product: {item['title']}\n"
                    f"Description: {item.get('description', 'N/A')}\n"
                    f"Price: ${item.get('price', 'N/A')}\n"
                    f"Relevance Score: {item.get('score', 'N/A')}\n"
                )
            elif item["type"] == "product":
                formatted.append(
                    f"Product: {item['data']['title']}\n"
                    f"Description: {item['data']['description']}\n"
                    f"Price: ${item['data']['price']}\n"
                )
            elif item["type"] == "review":
                formatted.append(
                    f"Review: {item['data']['text']}\n"
                    f"Rating: {item['data']['rating']}/5\n"
                    f"Helpful Votes: {item['data'].get('helpful_votes', 0)}\n"
                )
            elif item["type"] == "community_product":
                formatted.append(
                    f"Popular in Community: {item['data']['title']}\n"
                    f"Price: ${item['data']['price']}\n"
                )
        return "\n".join(formatted)

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

if __name__ == "__main__":
    # Example usage
    graph_rag = GraphRAG()
    try:
        # Example query with user context
        query = "What are the best wireless headphones under $100?"
        user_id = "user123"  # Optional user ID for community-aware recommendations
        answer, context = graph_rag.answer_query(query, user_id)
        print(f"Question: {query}")
        print(f"Answer: {answer}")
    finally:
        graph_rag.close() 