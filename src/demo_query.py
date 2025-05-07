import argparse
import json
from typing import Dict, Any
from retrieval.graph_rag import GraphRAG
from retrieval.semantic import SemanticRetriever
from evaluation.gpt_scorer import GPTScorer

def load_config(config_path: str = "config/evaluation_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_demo_query(
    query: str,
    query_type: str,
    config_path: str = "config/evaluation_config.yaml",
    use_gpt_scoring: bool = True
) -> Dict[str, Any]:
    """Run a single query through both retrieval methods and compare results."""
    # Load configuration
    config = load_config(config_path)
    
    # Initialize retrievers
    graph_rag = GraphRAG(config)
    semantic = SemanticRetriever(config)
    
    # Run retrievals
    print("\nRunning GraphRAG retrieval...")
    graph_rag_results = graph_rag.retrieve(query, query_type)
    
    print("\nRunning Semantic retrieval...")
    semantic_results = semantic.retrieve(query, query_type)
    
    # Format results
    results = {
        "query": query,
        "type": query_type,
        "graph_rag": {
            "retrieved_asins": graph_rag_results["retrieved_asins"],
            "context": graph_rag_results["context"],
            "answer": graph_rag_results.get("answer", None)
        },
        "semantic": {
            "retrieved_asins": semantic_results["retrieved_asins"],
            "context": semantic_results["context"],
            "answer": semantic_results.get("answer", None)
        }
    }
    
    # Run GPT scoring if requested
    if use_gpt_scoring:
        print("\nRunning GPT scoring...")
        gpt_scorer = GPTScorer()
        scored_results = gpt_scorer.score_single_query(results)
        results["gpt_scores"] = scored_results
    
    return results

def format_results(results: Dict[str, Any]) -> str:
    """Format results for display."""
    output = [
        f"\nQuery: {results['query']}",
        f"Type: {results['type']}\n",
        
        "GraphRAG Results:",
        "----------------",
        f"Retrieved ASINs: {', '.join(results['graph_rag']['retrieved_asins'])}",
        f"Context: {results['graph_rag']['context']}",
        f"Answer: {results['graph_rag']['answer']}\n",
        
        "Semantic Results:",
        "----------------",
        f"Retrieved ASINs: {', '.join(results['semantic']['retrieved_asins'])}",
        f"Context: {results['semantic']['context']}",
        f"Answer: {results['semantic']['answer']}\n"
    ]
    
    if "gpt_scores" in results:
        output.extend([
            "GPT Scoring Results:",
            "-------------------",
            "GraphRAG:",
            *[f"- {criterion}: {score:.3f}" 
              for criterion, score in results["gpt_scores"]["graph_rag"].items()],
            "\nSemantic:",
            *[f"- {criterion}: {score:.3f}" 
              for criterion, score in results["gpt_scores"]["semantic"].items()]
        ])
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Run a demo query through the retrieval system")
    parser.add_argument("query", help="The query to run")
    parser.add_argument("--type", choices=["relationship", "attribute"], 
                       default="relationship", help="Query type")
    parser.add_argument("--config", default="config/evaluation_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--no-gpt", action="store_true",
                       help="Disable GPT scoring")
    parser.add_argument("--output", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Run query
    results = run_demo_query(
        query=args.query,
        query_type=args.type,
        config_path=args.config,
        use_gpt_scoring=not args.no_gpt
    )
    
    # Print results
    print(format_results(results))
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 