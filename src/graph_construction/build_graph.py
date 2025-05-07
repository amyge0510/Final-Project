import os
import yaml
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from typing import Dict, List, Any

class GraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def create_constraints(self):
        """Create unique constraints for nodes."""
        with self.driver.session() as session:
            # Product constraints
            session.run("CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT product_asin IF NOT EXISTS FOR (p:Product) REQUIRE p.asin IS UNIQUE")
            
            # User constraints
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            
            # Category constraints
            session.run("CREATE CONSTRAINT category_id IF NOT EXISTS FOR (c:Category) REQUIRE c.id IS UNIQUE")
            
    def build_graph(self, data_dir: str, config: Dict[str, Any]):
        """Build the knowledge graph from processed data."""
        # Load processed data
        products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))
        users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        reviews_df = pd.read_csv(os.path.join(data_dir, 'reviews.csv'))
        categories_df = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
        
        # Create constraints
        self.create_constraints()
        
        # Create nodes and relationships
        self._create_product_nodes(products_df)
        self._create_user_nodes(users_df)
        self._create_category_nodes(categories_df)
        self._create_review_relationships(reviews_df, config)
        self._create_similar_product_relationships(products_df, config)
        self._create_category_relationships(products_df)
        
    def _create_product_nodes(self, products_df: pd.DataFrame):
        """Create product nodes in the graph."""
        with self.driver.session() as session:
            for _, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Creating product nodes"):
                session.run("""
                    MERGE (p:Product {id: $id})
                    SET p.asin = $asin,
                        p.title = $title,
                        p.group = $group,
                        p.salesrank = $salesrank
                """, dict(row))
                
    def _create_user_nodes(self, users_df: pd.DataFrame):
        """Create user nodes in the graph."""
        with self.driver.session() as session:
            for _, row in tqdm(users_df.iterrows(), total=len(users_df), desc="Creating user nodes"):
                session.run("""
                    MERGE (u:User {id: $user_id})
                """, dict(row))
                
    def _create_category_nodes(self, categories_df: pd.DataFrame):
        """Create category nodes in the graph."""
        with self.driver.session() as session:
            for _, row in tqdm(categories_df.iterrows(), total=len(categories_df), desc="Creating category nodes"):
                session.run("""
                    MERGE (c:Category {id: $category_id})
                """, dict(row))
                
    def _create_review_relationships(self, reviews_df: pd.DataFrame, config: Dict[str, Any]):
        """Create review relationships between users and products."""
        with self.driver.session() as session:
            for _, row in tqdm(reviews_df.iterrows(), total=len(reviews_df), desc="Creating review relationships"):
                if (row['votes'] >= config['min_helpful_votes'] and 
                    row['rating'] >= config['min_rating']):
                    session.run("""
                        MATCH (u:User {id: $customer_id})
                        MATCH (p:Product {id: $product_id})
                        MERGE (u)-[r:REVIEWED]->(p)
                        SET r.rating = $rating,
                            r.votes = $votes,
                            r.helpful = $helpful,
                            r.date = datetime($date)
                    """, dict(row))
                    
    def _create_similar_product_relationships(self, products_df: pd.DataFrame, config: Dict[str, Any]):
        """Create similar product relationships."""
        with self.driver.session() as session:
            for _, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Creating similar product relationships"):
                if 'similar' in row and isinstance(row['similar'], str):
                    similar_asins = eval(row['similar'])  # Convert string list to actual list
                    for similar_asin in similar_asins:
                        session.run("""
                            MATCH (p1:Product {asin: $asin})
                            MATCH (p2:Product {asin: $similar_asin})
                            MERGE (p1)-[r:SIMILAR_TO]->(p2)
                            SET r.similarity = $similarity
                        """, {
                            'asin': row['asin'],
                            'similar_asin': similar_asin,
                            'similarity': config['min_similarity_threshold']
                        })
                        
    def _create_category_relationships(self, products_df: pd.DataFrame):
        """Create category relationships for products."""
        with self.driver.session() as session:
            for _, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Creating category relationships"):
                if 'categories' in row and isinstance(row['categories'], str):
                    categories = eval(row['categories'])  # Convert string list to actual list
                    for category in categories:
                        session.run("""
                            MATCH (p:Product {id: $product_id})
                            MATCH (c:Category {id: $category_id})
                            MERGE (p)-[r:BELONGS_TO]->(c)
                        """, {
                            'product_id': row['id'],
                            'category_id': category['id']
                        })

def main():
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Load Neo4j configuration
    with open('config/neo4j_config.yaml', 'r') as f:
        neo4j_config = yaml.safe_load(f)
        
    # Initialize graph builder
    builder = GraphBuilder(
        uri=neo4j_config['uri'],
        user=neo4j_config['user'],
        password=neo4j_config['password']
    )
    
    try:
        # Build the graph
        builder.build_graph(config['processed_data_path'], config)
        print("Knowledge graph construction completed successfully!")
    finally:
        builder.close()

if __name__ == '__main__':
    main() 