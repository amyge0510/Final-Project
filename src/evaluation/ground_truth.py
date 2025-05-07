"""Ground truth generation for evaluation."""
import json
import os
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import yaml

class GroundTruthBuilder:
    def __init__(self, config_path: str):
        """Initialize the ground truth builder."""
        self.config = self._load_config(config_path)
        self.products_df = None
        self.reviews_df = None
        self.copurchase_df = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_data(self, products_path: str, reviews_path: str, copurchase_path: str):
        """Load data from CSV files."""
        self.products_df = pd.read_csv(products_path)
        self.reviews_df = pd.read_csv(reviews_path)
        self.copurchase_df = pd.read_csv(copurchase_path)

    def _build_relationship_ground_truth(self) -> List[Dict[str, Any]]:
        """Build ground truth for relationship-based queries."""
        queries = []

        # Co-purchase relationship queries
        min_copurchases = self.config["evaluation"]["relationship_queries"]["co_purchase"]["min_copurchases"]
        copurchase_groups = self.copurchase_df.groupby("source_asin").size()
        frequent_sources = copurchase_groups[copurchase_groups >= min_copurchases].index

        for source_asin in frequent_sources[:10]:  # Limit to 10 queries per type
            source_product = self.products_df[self.products_df["asin"] == source_asin].iloc[0]
            target_asins = self.copurchase_df[self.copurchase_df["source_asin"] == source_asin]["target_asin"].tolist()
            
            queries.append({
                "query": f"What products are frequently bought together with '{source_product['title']}'?",
                "type": "relationship",
                "subtype": "co_purchase",
                "expected_asins": target_asins,
                "metadata": {
                    "source_asin": source_asin,
                    "source_title": source_product["title"],
                    "source_category": source_product["category"]
                }
            })

        # Review pattern queries
        min_rating = self.config["evaluation"]["relationship_queries"]["review_pattern"]["min_rating"]
        high_rated_products = self.reviews_df[self.reviews_df["rating"] >= min_rating]["product_asin"].unique()

        for product_asin in high_rated_products[:10]:  # Limit to 10 queries per type
            product = self.products_df[self.products_df["asin"] == product_asin].iloc[0]
            similar_products = self.products_df[
                (self.products_df["category"] == product["category"]) &
                (self.products_df["asin"] != product_asin)
            ]["asin"].tolist()[:5]

            queries.append({
                "query": f"What products are similar to '{product['title']}' based on customer reviews?",
                "type": "relationship",
                "subtype": "review_pattern",
                "expected_asins": similar_products,
                "metadata": {
                    "source_asin": product_asin,
                    "source_title": product["title"],
                    "source_category": product["category"]
                }
            })

        return queries

    def _build_attribute_ground_truth(self) -> List[Dict[str, Any]]:
        """Build ground truth for attribute-based queries."""
        queries = []

        # Rating-based queries
        min_rating = self.config["evaluation"]["attribute_queries"]["rating"]["min_rating"]
        high_rated = self.reviews_df.groupby("product_asin")["rating"].mean()
        top_rated_asins = high_rated[high_rated >= min_rating].index.tolist()[:10]

        for category in self.products_df["category"].unique():
            category_products = self.products_df[
                (self.products_df["category"] == category) &
                (self.products_df["asin"].isin(top_rated_asins))
            ]
            if not category_products.empty:
                queries.append({
                    "query": f"What are the highest-rated products in the {category} category?",
                    "type": "attribute",
                    "subtype": "rating",
                    "expected_asins": category_products["asin"].tolist(),
                    "metadata": {
                        "category": category,
                        "expected_titles": category_products["title"].tolist()
                    }
                })

        # Bestseller queries
        max_salesrank = self.config["evaluation"]["attribute_queries"]["bestseller"]["max_salesrank"]
        bestsellers = self.products_df[self.products_df["salesrank"] <= max_salesrank]

        for category in bestsellers["category"].unique():
            category_bestsellers = bestsellers[bestsellers["category"] == category]
            if not category_bestsellers.empty:
                queries.append({
                    "query": f"What are the bestselling products in the {category} category?",
                    "type": "attribute",
                    "subtype": "bestseller",
                    "expected_asins": category_bestsellers["asin"].tolist()[:5],
                    "metadata": {
                        "category": category,
                        "expected_titles": category_bestsellers["title"].tolist()[:5]
                    }
                })

        return queries

    def build_ground_truth(self) -> Dict[str, Any]:
        """Build complete ground truth dataset."""
        relationship_queries = self._build_relationship_ground_truth()
        attribute_queries = self._build_attribute_ground_truth()

        return {
            "relationship_queries": relationship_queries,
            "attribute_queries": attribute_queries,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": self.config["evaluation"],
                "stats": {
                    "num_relationship_queries": len(relationship_queries),
                    "num_attribute_queries": len(attribute_queries)
                }
            }
        }

    def save_ground_truth(self, output_path: str):
        """Save ground truth to JSON file."""
        ground_truth = self.build_ground_truth()
        
        with open(output_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)

        print(f"Generated {ground_truth['metadata']['stats']['num_relationship_queries']} relationship queries")
        print(f"- {ground_truth['metadata']['stats']['num_attribute_queries']} attribute queries")

def main():
    # Example usage
    builder = GroundTruthBuilder(config_path="config/evaluation_config.yaml")
    builder.load_data(
        products_path="data/processed/products.csv",
        reviews_path="data/processed/reviews.csv",
        copurchase_path="data/processed/copurchase.csv"
    )
    builder.save_ground_truth("data/evaluation/ground_truth.json")

if __name__ == "__main__":
    main() 