import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

import numpy as np
import yaml
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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

    def _find_anchor_nodes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find initial anchor nodes using semantic search."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        result = tx.run(
                            """
                            CALL db.index.vector.queryNodes('product_embeddings', $top_k, $embedding)
                            YIELD node, score
                            RETURN node.asin as asin,
                                   node.title as title,
                                   node.category as category,
                                   node.description as description,
                                   node.price as price,
                                   score
                            """,
                            embedding=query_embedding,
                            top_k=top_k
                        )
                        return [dict(record) for record in result]
                    
            except Exception as e:
                error_str = str(e)
                if "insufficient_quota" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print("Max retries reached for rate limit")
                        raise
                else:
                    raise

    def _get_product_context(self, session, asin: str, max_hops: int) -> List[Dict[str, Any]]:
        """Get product context by traversing the graph."""
        with session.begin_transaction() as tx:
            result = tx.run(
                """
                MATCH (p:Product {asin: $asin})
                OPTIONAL MATCH (p)-[r:CO_PURCHASED_WITH*1..$max_hops]-(related:Product)
                RETURN p as product,
                       collect(distinct related) as related_products
                """,
                asin=asin,
                max_hops=max_hops
            )
            context = []
            for record in result:
                product = record["product"]
                context.append({
                    "type": "product",
                    "data": {
                        "asin": product["asin"],
                        "title": product["title"],
                        "category": product["category"],
                        "description": product.get("description", ""),
                        "price": product.get("price", 0.0)
                    }
                })
                for related in record["related_products"]:
                    if related:
                        context.append({
                            "type": "product",
                            "data": {
                                "asin": related["asin"],
                                "title": related["title"],
                                "category": related["category"],
                                "description": related.get("description", ""),
                                "price": related.get("price", 0.0)
                            }
                        })
            return context

    def _get_user_community(self, user_id: str) -> str:
        """Get user's community ID."""
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(
                    """
                    MATCH (u:User {id: $user_id})-[:BELONGS_TO]->(c:Community)
                    RETURN c.id as community_id
                    """,
                    user_id=user_id
                )
                record = result.single()
                return record["community_id"] if record else None

    def _get_community_context(self, session, asin: str, community_id: str, max_hops: int) -> List[Dict[str, Any]]:
        """Get community-specific context."""
        with session.begin_transaction() as tx:
            result = tx.run(
                """
                MATCH (p:Product {asin: $asin})
                MATCH (c:Community {id: $community_id})
                OPTIONAL MATCH (c)-[:HAS_MEMBER]->(u:User)-[r:REVIEWED]->(related:Product)
                WHERE r.rating >= 4
                RETURN collect(distinct related) as community_products
                """,
                asin=asin,
                community_id=community_id
            )
            context = []
            for record in result:
                for product in record["community_products"]:
                    if product:
                        context.append({
                            "type": "community_product",
                            "data": {
                                "asin": product["asin"],
                                "title": product["title"],
                                "category": product["category"],
                                "price": product.get("price", 0.0)
                            }
                        })
            return context

    def _deduplicate_context(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate products from context."""
        seen_asins = set()
        unique_context = []
        for item in context:
            asin = item["data"]["asin"] if "data" in item else item.get("asin")
            if asin and asin not in seen_asins:
                seen_asins.add(asin)
                unique_context.append(item)
        return unique_context

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
        return self._find_anchor_nodes(query, top_k)

    def answer_query(self, query: str, user_id: Optional[str] = None, method: str = "graph") -> Dict[str, Any]:
        """Answer a query using either GraphRAG or semantic retrieval."""
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
        
        answer = self.llm.invoke(prompt)
        return {
            "answer": answer,
            "context": context,
            "retrieved_asins": [item["data"]["asin"] if "data" in item else item["asin"] for item in context]
        }

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for the LLM."""
        formatted = []
        for item in context:
            if isinstance(item, dict) and 'title' in item:  # Semantic search result
                formatted.append(
                    f"Product: {item['title']}\n"
                    f"Category: {item.get('category', 'N/A')}\n"
                    f"Description: {item.get('description', 'N/A')}\n"
                    f"Price: ${item.get('price', 'N/A')}\n"
                    f"Relevance Score: {item.get('score', 'N/A')}\n"
                )
            elif item.get("type") == "product":
                formatted.append(
                    f"Product: {item['data']['title']}\n"
                    f"Category: {item['data']['category']}\n"
                    f"Description: {item['data']['description']}\n"
                    f"Price: ${item['data']['price']}\n"
                )
            elif item.get("type") == "review":
                formatted.append(
                    f"Review: {item['data']['text']}\n"
                    f"Rating: {item['data']['rating']}/5\n"
                    f"Helpful Votes: {item['data'].get('helpful_votes', 0)}\n"
                )
            elif item.get("type") == "community_product":
                formatted.append(
                    f"Popular in Community: {item['data']['title']}\n"
                    f"Category: {item['data']['category']}\n"
                    f"Price: ${item['data']['price']}\n"
                )
        return "\n".join(formatted)

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

if __name__ == "__main__":
    # Example usage
    graph_rag = GraphRAG()
    try:
        # Example query with user context
        query = "What are the best wireless headphones under $100?"
        user_id = "user123"  # Optional user ID for community-aware recommendations
        result = graph_rag.answer_query(query, user_id)
        print(f"Question: {query}")
        print(f"Answer: {result['answer']}")
    finally:
        graph_rag.close() 