import os
import yaml
import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, Any
from tqdm import tqdm

class CommunityDetector:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def detect_communities(self, config: Dict[str, Any]):
        """Detect user communities based on product interactions."""
        with self.driver.session() as session:
            # Create user similarity graph based on common product interactions
            print("Creating user similarity graph...")
            session.run("""
                CALL gds.graph.project(
                    'user_similarity',
                    ['User'],
                    {
                        SIMILAR_USERS: {
                            type: 'IN_COMMUNITY',
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
            """)
            
            # Calculate user similarities based on product interactions
            print("Calculating user similarities...")
            session.run("""
                MATCH (u1:User)-[r1:REVIEWED]->(p:Product)<-[r2:REVIEWED]-(u2:User)
                WHERE u1 <> u2 AND r1.rating >= $min_rating AND r2.rating >= $min_rating
                WITH u1, u2, COUNT(p) as common_products,
                     gds.similarity.cosine(
                         collect(r1.rating),
                         collect(r2.rating)
                     ) as similarity
                WHERE common_products >= $min_common_products
                MERGE (u1)-[s:IN_COMMUNITY]-(u2)
                SET s.similarity_score = similarity
            """, {
                'min_rating': config['min_rating'],
                'min_common_products': 2
            })
            
            # Run Louvain community detection
            print("Detecting communities...")
            session.run("""
                CALL gds.louvain.write(
                    'user_similarity',
                    {
                        writeProperty: 'community_id',
                        relationshipWeightProperty: 'similarity_score',
                        minCommunitySize: $min_size,
                        resolution: $resolution
                    }
                )
            """, {
                'min_size': config['community_min_size'],
                'resolution': config['community_resolution']
            })
            
            # Get community statistics
            result = session.run("""
                MATCH (u:User)
                WHERE u.community_id IS NOT NULL
                RETURN u.community_id as community_id,
                       COUNT(*) as size
                ORDER BY size DESC
            """)
            
            communities = []
            for record in result:
                communities.append({
                    'community_id': record['community_id'],
                    'size': record['size']
                })
                
            print(f"\nDetected {len(communities)} communities:")
            for comm in communities[:10]:
                print(f"Community {comm['community_id']}: {comm['size']} users")
                
            # Calculate community preferences
            print("\nCalculating community preferences...")
            session.run("""
                MATCH (u:User)-[r:REVIEWED]->(p:Product)
                WHERE u.community_id IS NOT NULL
                WITH u.community_id as community_id,
                     p.group as product_group,
                     COUNT(*) as purchase_count,
                     AVG(r.rating) as avg_rating
                WHERE purchase_count >= $min_reviews_per_product
                RETURN community_id,
                       product_group,
                       purchase_count,
                       avg_rating
                ORDER BY community_id, purchase_count DESC
            """, {
                'min_reviews_per_product': config['min_reviews_per_product']
            })
            
def main():
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Load Neo4j configuration
    with open('config/neo4j_config.yaml', 'r') as f:
        neo4j_config = yaml.safe_load(f)
        
    # Initialize community detector
    detector = CommunityDetector(
        uri=neo4j_config['uri'],
        user=neo4j_config['user'],
        password=neo4j_config['password']
    )
    
    try:
        # Detect communities
        detector.detect_communities(config)
        print("\nCommunity detection completed successfully!")
    finally:
        detector.close()

if __name__ == '__main__':
    main() 