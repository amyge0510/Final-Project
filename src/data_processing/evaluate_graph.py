import os
import json
import yaml
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from graph_constructor import GraphConstructor

class GraphEvaluator:
    def __init__(self, config_path: str = "config/evaluation_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.constructor = GraphConstructor()
        
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return metrics."""
        # Load data
        data_dir = "data/processed"
        self.constructor.load_data(data_dir)
        
        # Initialize results
        results = {
            'timestamp': datetime.now().strftime(self.config['output']['timestamp_format']),
            'metrics': {},
            'query_results': {}
        }
        
        # Evaluate relationship queries
        results['query_results']['relationship'] = self._evaluate_relationship_queries()
        
        # Evaluate attribute queries
        results['query_results']['attribute'] = self._evaluate_attribute_queries()
        
        # Calculate overall metrics
        results['metrics'] = self._calculate_metrics(results['query_results'])
        
        return results
    
    def _evaluate_relationship_queries(self) -> Dict[str, Any]:
        """Evaluate relationship-based queries, including multi-hop."""
        results = {}
        # Co-purchase based queries
        for product in self._get_all_products()[:10]:  # Limit for speed
            query = f"What books are frequently bought together with '{product['title']}'?"
            results[query] = {
                'query': query,
                'results': self.constructor.find_similar_products(product['asin'], max_hops=1),
                'expected': self._get_expected_co_purchases(product['asin'])
            }
        # Multi-hop relationship queries
        for user in self._get_all_users()[:10]:  # Limit for speed
            query = f"What books can be recommended to user {user} via multi-hop co-purchase?"
            results[query] = {
                'query': query,
                'results': self.constructor.multi_hop_recommendations(user, hops=2),
                'expected': []  # No ground truth for multi-hop, just show results
            }
        return results
    
    def _evaluate_attribute_queries(self) -> Dict[str, Any]:
        """Evaluate attribute-based queries, including advanced ones."""
        results = {}
        # Rating based queries
        query = "What are the highest rated books in the dataset?"
        results[query] = {
            'query': query,
            'results': self._get_highest_rated_products(),
            'expected': self._get_expected_highest_rated()
        }
        # Advanced attribute query
        adv_query = "Find books with at least 10 reviews and average rating >= 4.5."
        results[adv_query] = {
            'query': adv_query,
            'results': self.constructor.advanced_attribute_query(min_reviews=10, min_avg_rating=4.5),
            'expected': []  # No ground truth, just show results
        }
        return results
    
    def _calculate_metrics(self, query_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mean_reciprocal_rank': 0.0,
            'normalized_dcg': 0.0
        }
        
        total_queries = 0
        for query_type in query_results:
            for query, result in query_results[query_type].items():
                if not result['results'] or not result['expected']:
                    continue
                    
                total_queries += 1
                # Calculate precision and recall
                relevant = set(result['expected'])
                retrieved = set(r['asin'] for r in result['results'])
                
                if len(retrieved) > 0:
                    precision = len(relevant & retrieved) / len(retrieved)
                    metrics['precision'] += precision
                    
                if len(relevant) > 0:
                    recall = len(relevant & retrieved) / len(relevant)
                    metrics['recall'] += recall
                    
                # Calculate MRR
                for i, item in enumerate(result['results']):
                    if item['asin'] in relevant:
                        metrics['mean_reciprocal_rank'] += 1.0 / (i + 1)
                        break
                        
                # Calculate NDCG
                dcg = 0.0
                idcg = 0.0
                for i, item in enumerate(result['results']):
                    if item['asin'] in relevant:
                        dcg += 1.0 / (np.log2(i + 2))
                for i in range(min(len(relevant), len(result['results']))):
                    idcg += 1.0 / (np.log2(i + 2))
                if idcg > 0:
                    metrics['normalized_dcg'] += dcg / idcg
                    
        # Average the metrics
        if total_queries > 0:
            for metric in metrics:
                metrics[metric] /= total_queries
                
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
        return metrics
    
    def _get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products in the graph."""
        return [{'asin': n, **d} for n, d in self.constructor.G.nodes(data=True) 
                if d.get('type') == 'Product']
    
    def _get_all_users(self) -> List[str]:
        """Get all users in the graph."""
        return [n for n, d in self.constructor.G.nodes(data=True) 
                if d.get('type') == 'User']
    
    def _get_all_categories(self) -> List[str]:
        """Get all categories in the graph."""
        return [n for n, d in self.constructor.G.nodes(data=True) 
                if d.get('type') == 'Category']
    
    def _get_expected_co_purchases(self, product_asin: str) -> List[str]:
        """Get expected co-purchased products."""
        return [n for n in self.constructor.G.successors(product_asin) 
                if self.constructor.G[product_asin][n].get('type') == 'SIMILAR_TO']
    
    def _get_expected_user_recommendations(self, user_id: str) -> List[str]:
        """Get expected user recommendations."""
        reviewed = [n for n in self.constructor.G.successors(user_id) 
                   if self.constructor.G[user_id][n].get('type') == 'REVIEWED']
        recommendations = []
        for product in reviewed:
            recommendations.extend(self._get_expected_co_purchases(product))
        return list(set(recommendations))
    
    def _get_highest_rated_products(self) -> List[Dict[str, Any]]:
        """Get highest rated products."""
        products = []
        for node, data in self.constructor.G.nodes(data=True):
            if data.get('type') == 'Product':
                ratings = [self.constructor.G[pred][node].get('rating') 
                          for pred in self.constructor.G.predecessors(node)
                          if self.constructor.G[pred][node].get('type') == 'REVIEWED']
                if ratings:
                    products.append({
                        'asin': node,
                        'title': data.get('title'),
                        'avg_rating': sum(ratings) / len(ratings)
                    })
        return sorted(products, key=lambda x: x['avg_rating'], reverse=True)
    
    def _get_expected_highest_rated(self) -> List[str]:
        """Get expected highest rated products."""
        return [p['asin'] for p in self._get_highest_rated_products()]
    
    def _get_products_in_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products in a specific category."""
        products = []
        for node in self.constructor.G.predecessors(category):
            if self.constructor.G[node][category].get('type') == 'BELONGS_TO':
                products.append({
                    'asin': node,
                    'title': self.constructor.G.nodes[node].get('title')
                })
        return products
    
    def _get_expected_category_products(self, category: str) -> List[str]:
        """Get expected products in a category."""
        return [p['asin'] for p in self._get_products_in_category(category)]

def main():
    # Initialize evaluator
    evaluator = GraphEvaluator()
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate()
    
    # Create results directory if it doesn't exist
    os.makedirs("../../results/evaluation", exist_ok=True)
    
    # Save results
    output_file = os.path.join(
        "../../results/evaluation",
        f"evaluation_results_{results['timestamp']}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Print summary
    print("\nEvaluation Results Summary:")
    print(f"Results saved to: {output_file}")
    print("\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")
        
    # Print example query results
    print("\nExample Query Results:")
    for query_type, queries in results['query_results'].items():
        print(f"\n{query_type.upper()} Queries:")
        for query, result in list(queries.items())[:2]:  # Show first 2 queries of each type
            print(f"\nQuery: {query}")
            print("Top 3 Results:")
            for r in result['results'][:3]:
                print(f"- {r.get('title', r.get('asin'))}")

if __name__ == "__main__":
    main() 