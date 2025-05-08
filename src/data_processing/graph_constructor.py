import os
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np

class GraphConstructor:
    def __init__(self):
        self.G = nx.DiGraph()  # Directed graph for relationships
        self.metrics = {
            'products': 0,
            'users': 0,
            'categories': 0,
            'reviews': 0,
            'co_purchases': 0,
            'category_relationships': 0
        }
        
    def load_data(self, data_dir: str):
        """Load all CSV files and construct the graph."""
        print("Loading data from CSV files...")
        
        # Load products
        products_df = pd.read_csv(os.path.join(data_dir, "products.csv"))
        self._add_products(products_df)
        
        # Load users
        users_df = pd.read_csv(os.path.join(data_dir, "users.csv"))
        self._add_users(users_df)
        
        # Load categories
        categories_df = pd.read_csv(os.path.join(data_dir, "categories.csv"))
        self._add_categories(categories_df)
        
        # Load reviews
        reviews_df = pd.read_csv(os.path.join(data_dir, "reviews.csv"))
        self._add_reviews(reviews_df)
        
        # Load co-purchases
        co_purchases_df = pd.read_csv(os.path.join(data_dir, "co_purchases.csv"))
        self._add_co_purchases(co_purchases_df)
        
        # Add category relationships
        self._add_category_relationships(products_df)
        
    def _add_products(self, products_df: pd.DataFrame):
        """Add products as nodes to the graph."""
        for _, row in tqdm(products_df.iterrows(), desc="Adding products"):
            self.G.add_node(
                row['asin'],
                type='Product',
                title=row['title'],
                group=row['group'],
                salesrank=row['salesrank']
            )
            self.metrics['products'] += 1
            
    def _add_users(self, users_df: pd.DataFrame):
        """Add users as nodes to the graph."""
        for _, row in tqdm(users_df.iterrows(), desc="Adding users"):
            self.G.add_node(
                row['user_id'],
                type='User'
            )
            self.metrics['users'] += 1
            
    def _add_categories(self, categories_df: pd.DataFrame):
        """Add categories as nodes to the graph."""
        for _, row in tqdm(categories_df.iterrows(), desc="Adding categories"):
            self.G.add_node(
                row['category_id'],
                type='Category'
            )
            self.metrics['categories'] += 1
            
    def _add_reviews(self, reviews_df: pd.DataFrame):
        """Add reviews as edges between users and products."""
        for _, row in tqdm(reviews_df.iterrows(), desc="Adding reviews"):
            self.G.add_edge(
                row['customer_id'],
                row['product_asin'],
                type='REVIEWED',
                rating=row['rating'],
                votes=row['votes'],
                helpful=row['helpful'],
                date=row['date']
            )
            self.metrics['reviews'] += 1
            
    def _add_co_purchases(self, co_purchases_df: pd.DataFrame):
        """Add co-purchase relationships between products."""
        for _, row in tqdm(co_purchases_df.iterrows(), desc="Adding co-purchases"):
            self.G.add_edge(
                row['source'],
                row['target'],
                type='SIMILAR_TO',
                similarity=1.0
            )
            self.metrics['co_purchases'] += 1
            
    def _add_category_relationships(self, products_df: pd.DataFrame):
        """Add category relationships for products."""
        for _, row in tqdm(products_df.iterrows(), desc="Adding category relationships"):
            categories = eval(row['categories'])
            for category in categories:
                self.G.add_edge(
                    row['asin'],
                    category['id'],
                    type='BELONGS_TO'
                )
                self.metrics['category_relationships'] += 1
                
    def get_graph_stats(self) -> Dict[str, Any]:
        """Calculate and return graph statistics."""
        stats = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'metrics': self.metrics,
            'node_types': {
                'Product': len([n for n, d in self.G.nodes(data=True) if d.get('type') == 'Product']),
                'User': len([n for n, d in self.G.nodes(data=True) if d.get('type') == 'User']),
                'Category': len([n for n, d in self.G.nodes(data=True) if d.get('type') == 'Category'])
            },
            'edge_types': {
                'REVIEWED': len([e for e in self.G.edges(data=True) if e[2].get('type') == 'REVIEWED']),
                'SIMILAR_TO': len([e for e in self.G.edges(data=True) if e[2].get('type') == 'SIMILAR_TO']),
                'BELONGS_TO': len([e for e in self.G.edges(data=True) if e[2].get('type') == 'BELONGS_TO'])
            }
        }
        
        # Calculate average metrics
        if stats['node_types']['Product'] > 0:
            stats['avg_reviews_per_product'] = stats['edge_types']['REVIEWED'] / stats['node_types']['Product']
        if stats['node_types']['User'] > 0:
            stats['avg_reviews_per_user'] = stats['edge_types']['REVIEWED'] / stats['node_types']['User']
            
        return stats
    
    def find_similar_products(self, product_asin: str, max_hops: int = 2) -> list:
        """Find similar products based on co-purchases and shared categories."""
        similar_products = set()
        # 1-hop co-purchase
        for n in self.G.successors(product_asin):
            if self.G[product_asin][n].get('type') == 'SIMILAR_TO' and self.G.nodes[n].get('type') == 'Product':
                similar_products.add(n)
        # 2-hop co-purchase (multi-hop)
        if max_hops > 1:
            for n in list(similar_products):
                for m in self.G.successors(n):
                    if self.G[n][m].get('type') == 'SIMILAR_TO' and self.G.nodes[m].get('type') == 'Product':
                        similar_products.add(m)
        # Add category-based similarity
        product_categories = [n for n in self.G.successors(product_asin) if self.G[product_asin][n].get('type') == 'BELONGS_TO']
        for cat in product_categories:
            for n in self.G.predecessors(cat):
                if self.G[n][cat].get('type') == 'BELONGS_TO' and n != product_asin and self.G.nodes[n].get('type') == 'Product':
                    similar_products.add(n)
        # Format output
        results = []
        for asin in similar_products:
            node_data = self.G.nodes[asin]
            results.append({
                'asin': asin,
                'title': node_data.get('title', ''),
                'similarity_score': 1.0
            })
        return sorted(results, key=lambda x: x['title'])

    def multi_hop_recommendations(self, user_id: str, hops: int = 2) -> list:
        """Recommend products to a user by traversing multi-hop relationships (user->product->co-purchase->product)."""
        if user_id not in self.G:
            return []
        reviewed = [n for n in self.G.successors(user_id) if self.G[user_id][n].get('type') == 'REVIEWED']
        recs = set()
        for prod in reviewed:
            # 1-hop: co-purchase
            for n in self.G.successors(prod):
                if self.G[prod][n].get('type') == 'SIMILAR_TO' and self.G.nodes[n].get('type') == 'Product':
                    recs.add(n)
            # 2-hop: co-purchase of co-purchase
            if hops > 1:
                for n in self.G.successors(prod):
                    if self.G[prod][n].get('type') == 'SIMILAR_TO' and self.G.nodes[n].get('type') == 'Product':
                        for m in self.G.successors(n):
                            if self.G[n][m].get('type') == 'SIMILAR_TO' and self.G.nodes[m].get('type') == 'Product':
                                recs.add(m)
        # Remove already reviewed
        recs = recs - set(reviewed)
        return [{
            'asin': asin,
            'title': self.G.nodes[asin].get('title', '')
        } for asin in recs]

    def advanced_attribute_query(self, min_reviews=2, min_avg_rating=4.0) -> list:
        """Find products with at least min_reviews and average rating >= min_avg_rating."""
        results = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'Product':
                ratings = [self.G[pred][node].get('rating') for pred in self.G.predecessors(node) if self.G[pred][node].get('type') == 'REVIEWED']
                if len(ratings) >= min_reviews and np.mean(ratings) >= min_avg_rating:
                    results.append({
                        'asin': node,
                        'title': data.get('title', ''),
                        'avg_rating': np.mean(ratings),
                        'num_reviews': len(ratings)
                    })
        return sorted(results, key=lambda x: (-x['avg_rating'], -x['num_reviews']))

    def detect_communities(self):
        """Detect communities in the product graph using label propagation."""
        communities = list(nx.algorithms.community.label_propagation_communities(self.G.to_undirected()))
        self.node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                self.node_community[node] = i
        return self.node_community

    def community_recommendations(self, user_id: str, top_k: int = 5):
        """Recommend products from the user's community that they haven't reviewed yet."""
        if not hasattr(self, 'node_community'):
            self.detect_communities()
        # Find products reviewed by user
        reviewed = set(n for n in self.G.successors(user_id) if self.G[user_id][n].get('type') == 'REVIEWED')
        # Find user's community (based on reviewed products)
        user_communities = set(self.node_community.get(prod) for prod in reviewed if prod in self.node_community)
        # Recommend top products from these communities not yet reviewed
        candidates = [n for n, d in self.G.nodes(data=True)
                      if d.get('type') == 'Product' and self.node_community.get(n) in user_communities and n not in reviewed]
        # Rank by number of reviews
        ranked = sorted(candidates, key=lambda x: len([p for p in self.G.predecessors(x) if self.G[p][x].get('type') == 'REVIEWED']), reverse=True)
        return [{'asin': n, 'title': self.G.nodes[n].get('title', '')} for n in ranked[:top_k]]

    def two_hop_also_bought(self, product_asin: str, top_k: int = 5):
        """Find products that are 2-hops away via co-purchase (also-bought-of-similar)."""
        one_hop = set(n for n in self.G.successors(product_asin) if self.G[product_asin][n].get('type') == 'SIMILAR_TO')
        two_hop = set()
        for n in one_hop:
            two_hop.update(m for m in self.G.successors(n) if self.G[n][m].get('type') == 'SIMILAR_TO')
        two_hop -= one_hop
        two_hop.discard(product_asin)
        ranked = sorted(two_hop, key=lambda x: len([p for p in self.G.predecessors(x) if self.G[p][x].get('type') == 'REVIEWED']), reverse=True)
        return [{'asin': n, 'title': self.G.nodes[n].get('title', '')} for n in ranked[:top_k]]

def main():
    # Initialize graph constructor
    constructor = GraphConstructor()
    
    # Load data
    data_dir = "data/processed"
    constructor.load_data(data_dir)
    
    # Get and print graph statistics
    stats = constructor.get_graph_stats()
    print("\nGraph Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Example queries
    print("\nExample Queries:")
    
    # Find similar products for the first product
    first_product = next((n for n, d in constructor.G.nodes(data=True) if d.get('type') == 'Product'), None)
    if first_product:
        product_data = constructor.G.nodes[first_product]
        print(f"\nSimilar products to {product_data.get('title')}:")
        similar = constructor.find_similar_products(first_product)
        for prod in similar:
            print(f"- {prod['title']} (score: {prod['similarity_score']:.2f})")
    
    # Get recommendations for the first user
    first_user = next((n for n, d in constructor.G.nodes(data=True) if d.get('type') == 'User'), None)
    if first_user:
        print(f"\nRecommendations for user {first_user}:")
        recommendations = constructor.multi_hop_recommendations(first_user)
        for rec in recommendations:
            print(f"- {rec['title']}")
            
    # Print all products and their relationships
    print("\nAll Products and Their Relationships:")
    for node, data in constructor.G.nodes(data=True):
        if data.get('type') == 'Product':
            print(f"\nProduct: {data.get('title')}")
            # Get co-purchased products
            co_purchased = [n for n in constructor.G.successors(node) 
                          if constructor.G[node][n].get('type') == 'SIMILAR_TO']
            if co_purchased:
                print("  Co-purchased with:")
                for cp in co_purchased:
                    print(f"  - {constructor.G.nodes[cp].get('title')}")
            # Get categories
            categories = [n for n in constructor.G.successors(node) 
                        if constructor.G[node][n].get('type') == 'BELONGS_TO']
            if categories:
                print("  Categories:")
                for cat in categories:
                    print(f"  - {cat}")
            # Get reviewers
            reviewers = [n for n in constructor.G.predecessors(node) 
                       if constructor.G[n][node].get('type') == 'REVIEWED']
            if reviewers:
                print("  Reviewed by:")
                for rev in reviewers:
                    rating = constructor.G[rev][node].get('rating')
                    print(f"  - User {rev} (rating: {rating})")

if __name__ == "__main__":
    main() 