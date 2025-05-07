import os
import yaml
from datetime import datetime
from typing import List, Dict, Any
import json
import numpy as np
from tabulate import tabulate

from .evaluator import QueryEvaluator
from .ground_truth import GroundTruthBuilder

def load_queries(config_path: str) -> List[Dict[str, Any]]:
    """Load queries from configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    queries = []
    for query_type, examples in config['query_types'].items():
        for query_text in examples:
            queries.append({
                'text': query_text,
                'type': query_type,
                'ground_truth': []  # This would be populated with actual ground truth data
            })
            
    return queries

def run_evaluation():
    """Run the complete evaluation pipeline."""
    # Load configuration
    config_path = "config/evaluation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Build ground truth if it doesn't exist
    ground_truth_path = 'data/evaluation/ground_truth.json'
    if not os.path.exists(ground_truth_path):
        print("Building ground truth dataset...")
        builder = GroundTruthBuilder('data/processed', config_path)
        builder.save_ground_truth(ground_truth_path)
    
    # Step 2: Run evaluation
    print("\nRunning evaluation...")
    evaluator = QueryEvaluator(config_path)
    
    try:
        results = evaluator.evaluate_queries()
        
        # Step 3: Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            "results/evaluation",
            f"evaluation_results_{timestamp}.json"
        )
        evaluator.save_results(results, output_path)
        
        # Step 4: Print analysis
        print("\nEvaluation Results Analysis")
        print("=========================")
        
        # Create comparison table
        table_data = []
        for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
            for query_type in ["relationship", "attribute"]:
                graph_rag = results["graph_rag"][query_type]
                semantic = results["semantic"][query_type]
                
                # Calculate improvement
                improvement = (
                    (graph_rag[f"mean_{metric}"] - semantic[f"mean_{metric}"]) /
                    semantic[f"mean_{metric}"] * 100
                )
                
                table_data.append([
                    metric.upper(),
                    query_type.title(),
                    f"{graph_rag[f'mean_{metric}']:.3f} ± {graph_rag[f'std_{metric}']:.3f}",
                    f"{semantic[f'mean_{metric}']:.3f} ± {semantic[f'std_{metric}']:.3f}",
                    f"{improvement:+.1f}%"
                ])
        
        # Print comparison table
        print("\nPerformance Comparison (GraphRAG vs Semantic)")
        print(tabulate(
            table_data,
            headers=["Metric", "Query Type", "GraphRAG", "Semantic", "Improvement"],
            tablefmt="grid"
        ))
        
        # Print statistical significance
        print("\nStatistical Significance Analysis")
        print("===============================")
        for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
            for query_type in ["relationship", "attribute"]:
                graph_rag = results["graph_rag"][query_type]
                semantic = results["semantic"][query_type]
                
                # Calculate t-statistic (assuming equal variances)
                n = len(graph_rag["queries"])  # Number of queries
                t_stat = (
                    (graph_rag[f"mean_{metric}"] - semantic[f"mean_{metric}"]) /
                    np.sqrt(
                        (graph_rag[f"std_{metric}"]**2 + semantic[f"std_{metric}"]**2) / n
                    )
                )
                
                # Print significance level
                significance = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else "ns"
                print(f"{metric.upper()} ({query_type}): t = {t_stat:.2f} {significance}")
        
        # Print summary of findings
        print("\nKey Findings")
        print("============")
        for query_type in ["relationship", "attribute"]:
            graph_rag = results["graph_rag"][query_type]
            semantic = results["semantic"][query_type]
            
            # Calculate average improvement across metrics
            improvements = []
            for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                improvement = (
                    (graph_rag[f"mean_{metric}"] - semantic[f"mean_{metric}"]) /
                    semantic[f"mean_{metric}"] * 100
                )
                improvements.append(improvement)
            
            avg_improvement = np.mean(improvements)
            print(f"\n{query_type.title()} Queries:")
            print(f"Average improvement: {avg_improvement:+.1f}%")
            print(f"Best performing metric: {max(improvements):+.1f}%")
            print(f"Worst performing metric: {min(improvements):+.1f}%")
            
    finally:
        evaluator.close()

if __name__ == "__main__":
    run_evaluation() 