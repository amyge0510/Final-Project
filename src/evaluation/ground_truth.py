import os
import json
import yaml
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

class GroundTruthBuilder:
    def __init__(self, data_dir: str, config_path: str = "config/evaluation_config.yaml"):
        self.data_dir = data_dir
        self.config = self._load_config(config_path)
        self.products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))
        self.reviews_df = pd.read_csv(os.path.join(data_dir, 'reviews.csv'))
        self.co_purchases_df = pd.read_csv(os.path.join(data_dir, 'co_purchases.csv'))
        self.review_relationships_df = pd.read_csv(os.path.join(data_dir, 'review_relationships.csv'))
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def build_ground_truth(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build ground truth data for each query type."""
        ground_truth = {
            "relationship": [],
            "attribute": []
        }
        
        # Build relationship query ground truth
        self._build_relationship_ground_truth(ground_truth)
        
        # Build attribute query ground truth
        self._build_attribute_ground_truth(ground_truth)
        
        return ground_truth
        
    def _build_relationship_ground_truth(self, ground_truth: Dict[str, List[Dict[str, Any]]]):
        """Build ground truth for relationship-based queries."""
        # Co-purchase based queries
        popular_books = self.products_df.nlargest(5, 'salesrank')
        for _, book in popular_books.iterrows():
            co_purchased = self.co_purchases_df[
                self.co_purchases_df['source'] == book['asin']
            ]['target'].tolist()
            
            if co_purchased:
                ground_truth["relationship"].append({
                    "text": f"What books are frequently bought together with '{book['title']}'?",
                    "type": "relationship",
                    "ground_truth": co_purchased[:5]  # Top 5 co-purchased books
                })
        
        # Review-based relationship queries
        highly_reviewed_books = self.products_df[
            self.products_df['review_stats'].apply(lambda x: x['total'] > 100)
        ].nlargest(5, 'review_stats.avg_rating')
        
        for _, book in highly_reviewed_books.iterrows():
            # Find books reviewed by the same users
            book_reviewers = self.review_relationships_df[
                self.review_relationships_df['product_asin'] == book['asin']
            ]['user_id'].unique()
            
            related_books = self.review_relationships_df[
                self.review_relationships_df['user_id'].isin(book_reviewers)
            ]['product_asin'].value_counts().head(5).index.tolist()
            
            if related_books:
                ground_truth["relationship"].append({
                    "text": f"What books do readers who gave 5-star reviews to '{book['title']}' also highly rate?",
                    "type": "relationship",
                    "ground_truth": related_books
                })
                
    def _build_attribute_ground_truth(self, ground_truth: Dict[str, List[Dict[str, Any]]]):
        """Build ground truth for attribute-based queries."""
        # Title and description based queries
        mystery_books = self.products_df[
            self.products_df['title'].str.contains('mystery', case=False)
        ].nlargest(5, 'salesrank')
        
        if not mystery_books.empty:
            ground_truth["attribute"].append({
                "text": "Find books with 'mystery' in the title",
                "type": "attribute",
                "ground_truth": mystery_books['asin'].tolist()
            })
            
        # Rating based queries
        highly_rated_books = self.products_df[
            self.products_df['review_stats'].apply(lambda x: x['avg_rating'] >= 4.5)
        ].nlargest(5, 'review_stats.total')
        
        if not highly_rated_books.empty:
            ground_truth["attribute"].append({
                "text": "Find books with at least 4.5 star rating and over 1000 reviews",
                "type": "attribute",
                "ground_truth": highly_rated_books['asin'].tolist()
            })
            
    def save_ground_truth(self, output_path: str):
        """Save ground truth data to JSON file."""
        ground_truth = self.build_ground_truth()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
            
        print(f"Saved ground truth data with:")
        print(f"- {len(ground_truth['relationship'])} relationship queries")
        print(f"- {len(ground_truth['attribute'])} attribute queries")

def main():
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Build and save ground truth
    builder = GroundTruthBuilder(config['processed_data_path'])
    builder.save_ground_truth('data/evaluation/ground_truth.json')

if __name__ == '__main__':
    main() 