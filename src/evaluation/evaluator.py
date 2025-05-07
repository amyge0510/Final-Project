import os
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import yaml

from ..retrieval.graph_rag import GraphRAG

class QueryEvaluator:
    def __init__(self, config_path: str = "config/evaluation_config.yaml"):
        self.graph_rag = GraphRAG()
        self.config = self._load_config(config_path)
        self.ground_truth = self._load_ground_truth()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_ground_truth(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load ground truth data."""
        with open('data/evaluation/ground_truth.json', 'r') as f:
            return json.load(f)
            
    def evaluate_queries(self) -> Dict[str, Any]:
        """Evaluate both GraphRAG and semantic retrieval on ground truth queries."""
        results = {
            "graph_rag": {"relationship": [], "attribute": []},
            "semantic": {"relationship": [], "attribute": []}
        }
        
        # Combine queries from both types
        all_queries = (
            self.ground_truth["relationship"] +
            self.ground_truth["attribute"]
        )
        
        for query in tqdm(all_queries, desc="Evaluating queries"):
            # Evaluate GraphRAG
            graph_answer, graph_context = self.graph_rag.answer_query(
                query["text"],
                method="graph"
            )
            
            # Verify ASINs in graph context
            graph_asins = self._extract_asins(graph_context)
            if not graph_asins:
                print(f"Warning: No ASINs found in graph context for query: {query['text']}")
            
            graph_metrics = self._calculate_metrics(
                graph_context,
                query["ground_truth"]
            )
            results["graph_rag"][query["type"]].append({
                "query": query["text"],
                "metrics": graph_metrics,
                "retrieved_asins": graph_asins
            })
            
            # Evaluate semantic retrieval
            semantic_answer, semantic_context = self.graph_rag.answer_query(
                query["text"],
                method="semantic"
            )
            
            # Verify ASINs in semantic context
            semantic_asins = self._extract_asins(semantic_context)
            if not semantic_asins:
                print(f"Warning: No ASINs found in semantic context for query: {query['text']}")
            
            semantic_metrics = self._calculate_metrics(
                semantic_context,
                query["ground_truth"]
            )
            results["semantic"][query["type"]].append({
                "query": query["text"],
                "metrics": semantic_metrics,
                "retrieved_asins": semantic_asins
            })
            
        return self._aggregate_results(results)
        
    def _extract_asins(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract ASINs from retrieved context."""
        asins = set()
        for item in context:
            if isinstance(item, dict) and 'asin' in item:
                asins.add(item['asin'])
            elif isinstance(item, dict) and 'data' in item and 'asin' in item['data']:
                asins.add(item['data']['asin'])
        return list(asins)
        
    def _calculate_metrics(
        self,
        retrieved_context: List[Dict[str, Any]],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        retrieved_ids = self._extract_asins(retrieved_context)
        ground_truth_set = set(ground_truth)
        
        # Calculate precision and recall
        if not retrieved_ids:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0
            }
            
        true_positives = len(set(retrieved_ids).intersection(ground_truth_set))
        precision = true_positives / len(retrieved_ids)
        recall = true_positives / len(ground_truth_set)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate MRR
        mrr = 0.0
        for i, asin in enumerate(retrieved_ids):
            if asin in ground_truth_set:
                mrr = 1.0 / (i + 1)
                break
                
        # Calculate NDCG
        dcg = 0.0
        idcg = 0.0
        for i, asin in enumerate(retrieved_ids):
            if asin in ground_truth_set:
                dcg += 1.0 / np.log2(i + 2)
        for i in range(min(len(ground_truth_set), len(retrieved_ids))):
            idcg += 1.0 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg
        }
        
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evaluation results across query types."""
        aggregated = {}
        
        for method in ["graph_rag", "semantic"]:
            aggregated[method] = {}
            for query_type in ["relationship", "attribute"]:
                query_results = results[method][query_type]
                metrics = [r["metrics"] for r in query_results]
                
                aggregated[method][query_type] = {
                    "mean_precision": np.mean([m["precision"] for m in metrics]),
                    "mean_recall": np.mean([m["recall"] for m in metrics]),
                    "mean_f1": np.mean([m["f1"] for m in metrics]),
                    "mean_mrr": np.mean([m["mrr"] for m in metrics]),
                    "mean_ndcg": np.mean([m["ndcg"] for m in metrics]),
                    "std_precision": np.std([m["precision"] for m in metrics]),
                    "std_recall": np.std([m["recall"] for m in metrics]),
                    "std_f1": np.std([m["f1"] for m in metrics]),
                    "std_mrr": np.std([m["mrr"] for m in metrics]),
                    "std_ndcg": np.std([m["ndcg"] for m in metrics])
                }
                
        return aggregated
        
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
            
    def close(self):
        """Close the GraphRAG connection."""
        self.graph_rag.close() 