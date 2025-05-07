import os
import json
import yaml
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class GroundTruthBuilder:
    """Class for building ground truth data for evaluation."""
    
    def __init__(self, config_path: str = "config/evaluation_config.yaml"):
        """Initialize the ground truth builder."""
        self.config = self._load_config(config_path)
        self.products_df = None
        self.reviews_df = None
        self.copurchase_df = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, products_path: str, reviews_path: str, copurchase_path: str):
        """Load required data files."""
        self.products_df = pd.read_csv(products_path)
        self.reviews_df = pd.read_csv(reviews_path)
        self.copurchase_df = pd.read_csv(copurchase_path)
        
        # Ensure ASINs are strings
        self.products_df['asin'] = self.products_df['asin'].astype(str)
        self.reviews_df['product_asin'] = self.reviews_df['product_asin'].astype(str)
        self.copurchase_df['source_asin'] = self.copurchase_df['source_asin'].astype(str)
        self.copurchase_df['target_asin'] = self.copurchase_df['target_asin'].astype(str)
    
    def _build_relationship_ground_truth(self) -> List[Dict[str, Any]]:
        """Build ground truth for relationship queries."""
        relationship_queries = []
        
        # Get top products by number of co-purchases
        top_products = self.copurchase_df['source_asin'].value_counts().head(20).index
        
        for asin in top_products:
            # Get co-purchased products
            copurchased = self.copurchase_df[
                self.copurchase_df['source_asin'] == asin
            ]['target_asin'].tolist()
            
            # Get product details
            product = self.products_df[self.products_df['asin'] == asin].iloc[0]
            
            # Create co-purchase query
            query = {
                "query": f"What books are frequently bought together with '{product['title']}'?",
                "type": "relationship",
                "subtype": "co_purchase",
                "source_asin": asin,
                "expected_asins": copurchased[:5],  # Top 5 co-purchased items
                "metadata": {
                    "source_title": product['title'],
                    "source_category": product['category'],
                    "num_copurchases": len(copurchased)
                }
            }
            relationship_queries.append(query)
            
            # Create review-based query
            # Get products with similar review patterns
            similar_reviews = self.reviews_df[
                (self.reviews_df['product_asin'].isin(copurchased)) &
                (self.reviews_df['rating'] >= 4.0)
            ]['product_asin'].value_counts().head(5).index.tolist()
            
            query = {
                "query": f"What books are highly rated by readers who enjoyed '{product['title']}'?",
                "type": "relationship",
                "subtype": "review_pattern",
                "source_asin": asin,
                "expected_asins": similar_reviews,
                "metadata": {
                    "source_title": product['title'],
                    "source_category": product['category'],
                    "min_rating": 4.0
                }
            }
            relationship_queries.append(query)
        
        return relationship_queries
    
    def _build_attribute_ground_truth(self) -> List[Dict[str, Any]]:
        """Build ground truth for attribute queries."""
        attribute_queries = []
        
        # Get top categories
        top_categories = self.products_df['category'].value_counts().head(5).index
        
        for category in top_categories:
            # Get top-rated books in category
            top_rated = self.products_df[
                (self.products_df['category'] == category) &
                (self.products_df['rating'] >= 4.5)
            ].sort_values('rating', ascending=False).head(5)
            
            query = {
                "query": f"What are the highest-rated books in the {category} category?",
                "type": "attribute",
                "subtype": "rating",
                "expected_asins": top_rated['asin'].tolist(),
                "metadata": {
                    "category": category,
                    "min_rating": 4.5,
                    "expected_titles": top_rated['title'].tolist()
                }
            }
            attribute_queries.append(query)
            
            # Get recent bestsellers
            recent_bestsellers = self.products_df[
                (self.products_df['category'] == category) &
                (self.products_df['salesrank'] <= 1000)
            ].sort_values('salesrank').head(5)
            
            query = {
                "query": f"What are the current bestsellers in {category}?",
                "type": "attribute",
                "subtype": "bestseller",
                "expected_asins": recent_bestsellers['asin'].tolist(),
                "metadata": {
                    "category": category,
                    "max_salesrank": 1000,
                    "expected_titles": recent_bestsellers['title'].tolist()
                }
            }
            attribute_queries.append(query)
        
        return attribute_queries
    
    def build_ground_truth(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build complete ground truth dataset."""
        if self.products_df is None or self.reviews_df is None or self.copurchase_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        relationship_queries = self._build_relationship_ground_truth()
        attribute_queries = self._build_attribute_ground_truth()
        
        return {
            "relationship_queries": relationship_queries,
            "attribute_queries": attribute_queries,
            "metadata": {
                "total_queries": len(relationship_queries) + len(attribute_queries),
                "relationship_queries": len(relationship_queries),
                "attribute_queries": len(attribute_queries),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def save_ground_truth(self, output_path: str = "data/evaluation/ground_truth.json"):
        """Save ground truth data to JSON file."""
        ground_truth = self.build_ground_truth()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Ground truth data saved to {output_path}")
        print(f"Generated {ground_truth['metadata']['total_queries']} queries:")
        print(f"- {ground_truth['metadata']['relationship_queries']} relationship queries")
        print(f"- {ground_truth['metadata']['attribute_queries']} attribute queries")

def main():
    # Example usage
    builder = GroundTruthBuilder()
    builder.load_data(
        products_path="data/processed/products.csv",
        reviews_path="data/processed/reviews.csv",
        copurchase_path="data/processed/copurchase.csv"
    )
    builder.save_ground_truth()

if __name__ == "__main__":
    main() 