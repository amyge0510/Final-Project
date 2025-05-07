import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats
from tqdm import tqdm

from src.evaluation.evaluator import QueryEvaluator
from src.evaluation.statistical_analysis import StatisticalAnalyzer
from src.evaluation.gpt_scorer import GPTScorer

def load_config(config_path: str) -> dict:
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_results_directory() -> str:
    """Create results directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir / f"evaluation_results_{timestamp}.json")

def plot_performance_comparison(results: dict, output_dir: str):
    """Create visualizations comparing GraphRAG and semantic retrieval performance."""
    # Prepare data for plotting
    plot_data = []
    for method in ["graph_rag", "semantic"]:
        for query_type in ["relationship", "attribute"]:
            metrics = results["results"][method][f"{query_type}_metrics"]
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    plot_data.append({
                        "Method": method.upper(),
                        "Query Type": query_type.title(),
                        "Metric": metric_name.replace("_", " ").title(),
                        "Value": value
                    })
    
    df = pd.DataFrame(plot_data)
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Bar plot comparing methods across metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Method", ci=None)
    plt.title("Performance Comparison: GraphRAG vs Semantic Retrieval")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "performance_comparison.png"))
    plt.close()
    
    # Plot 2: Faceted plot by query type
    g = sns.FacetGrid(df, col="Query Type", height=6, aspect=1)
    g.map_dataframe(sns.barplot, x="Metric", y="Value", hue="Method", ci=None)
    g.add_legend()
    plt.xticks(rotation=45)
    g.fig.suptitle("Performance by Query Type", y=1.05)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "performance_by_query_type.png"))
    plt.close()

def plot_gpt_scores(gpt_results: dict, output_dir: str):
    """Create visualizations for GPT-4 scoring results."""
    plot_data = []
    for method in ["graph_rag", "semantic"]:
        for query_type in ["relationship", "attribute"]:
            scores = gpt_results["gpt_scores"][method][f"{query_type}_aggregate"]
            for criterion, score in scores.items():
                plot_data.append({
                    "Method": method.upper(),
                    "Query Type": query_type.title(),
                    "Criterion": criterion.title(),
                    "Score": score
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot GPT-4 scores comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Criterion", y="Score", hue="Method", ci=None)
    plt.title("GPT-4 Scoring: GraphRAG vs Semantic Retrieval")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "gpt_scores_comparison.png"))
    plt.close()
    
    # Plot GPT-4 scores by query type
    g = sns.FacetGrid(df, col="Query Type", height=6, aspect=1)
    g.map_dataframe(sns.barplot, x="Criterion", y="Score", hue="Method", ci=None)
    g.add_legend()
    plt.xticks(rotation=45)
    g.fig.suptitle("GPT-4 Scores by Query Type", y=1.05)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "gpt_scores_by_query_type.png"))
    plt.close()

def generate_summary_report(results: dict, gpt_results: dict, stats_results: dict, output_dir: str):
    """Generate a comprehensive summary report of all results."""
    report = []
    
    # Basic metrics summary
    report.append("# Evaluation Results Summary")
    report.append("\n## Performance Metrics")
    
    metrics_table = []
    headers = ["Method", "Query Type", "MRR", "NDCG", "Precision@k", "Recall@k"]
    for method in ["graph_rag", "semantic"]:
        for query_type in ["relationship", "attribute"]:
            metrics = results["results"][method][f"{query_type}_metrics"]
            metrics_table.append([
                method.upper(),
                query_type.title(),
                f"{metrics['mrr']:.3f}",
                f"{metrics['ndcg']:.3f}",
                f"{metrics['precision_at_k']:.3f}",
                f"{metrics['recall_at_k']:.3f}"
            ])
    
    report.append("\n" + tabulate(metrics_table, headers=headers, tablefmt="pipe"))
    
    # GPT-4 Scoring summary
    report.append("\n## GPT-4 Scoring Results")
    gpt_table = []
    headers = ["Method", "Query Type", "Relevance", "Completeness", "Accuracy", "Coherence", "Use of Context"]
    for method in ["graph_rag", "semantic"]:
        for query_type in ["relationship", "attribute"]:
            scores = gpt_results["gpt_scores"][method][f"{query_type}_aggregate"]
            gpt_table.append([
                method.upper(),
                query_type.title(),
                f"{scores['relevance']:.2f}",
                f"{scores['completeness']:.2f}",
                f"{scores['accuracy']:.2f}",
                f"{scores['coherence']:.2f}",
                f"{scores['use of context']:.2f}"
            ])
    
    report.append("\n" + tabulate(gpt_table, headers=headers, tablefmt="pipe"))
    
    # Statistical Analysis summary
    report.append("\n## Statistical Analysis")
    report.append("\n### ANOVA Results")
    report.append(f"F-statistic: {stats_results['anova']['f_statistic']:.3f}")
    report.append(f"p-value: {stats_results['anova']['p_value']:.3f}")
    report.append(f"Effect size (Eta-squared): {stats_results['anova']['effect_size']:.3f}")
    
    report.append("\n### Linear Regression Results")
    report.append(f"R-squared: {stats_results['regression']['r_squared']:.3f}")
    report.append("Feature Importance:")
    for feature, importance in stats_results['regression']['feature_importance'].items():
        report.append(f"- {feature}: {importance:.3f}")
    
    report.append("\n### Clustering Results")
    report.append("Cluster Characteristics:")
    for cluster, chars in stats_results['clustering']['cluster_characteristics'].items():
        report.append(f"\nCluster {cluster}:")
        for key, value in chars.items():
            report.append(f"- {key}: {value}")
    
    # Save report
    report_path = Path(output_dir) / "evaluation_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

def run_evaluation(
    config_path: str = "config/evaluation_config.yaml",
    ground_truth_path: str = "data/evaluation/ground_truth.json",
    output_dir: str = "results/evaluation"
) -> None:
    """Run the complete evaluation pipeline."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    evaluator = QueryEvaluator(config)
    analyzer = StatisticalAnalyzer()
    gpt_scorer = GPTScorer()
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Combine queries
    all_queries = (
        ground_truth["relationship_queries"] +
        ground_truth["attribute_queries"]
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_queries(all_queries)
    
    # Save raw results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save per-query results
    per_query_path = os.path.join(output_dir, "per_query_results.jsonl")
    with open(per_query_path, 'w') as f:
        for query in all_queries:
            query_results = {
                "query": query["query"],
                "type": query["type"],
                "expected_asins": query["expected_asins"],
                "graph_rag": {
                    "retrieved_asins": results["results"]["graph_rag"][query["type"]][
                        next(i for i, q in enumerate(all_queries) if q["query"] == query["query"])
                    ]["retrieved_asins"],
                    "metrics": results["results"]["graph_rag"][f"{query['type']}_metrics"]
                },
                "semantic": {
                    "retrieved_asins": results["results"]["semantic"][query["type"]][
                        next(i for i, q in enumerate(all_queries) if q["query"] == query["query"])
                    ]["retrieved_asins"],
                    "metrics": results["results"]["semantic"][f"{query['type']}_metrics"]
                }
            }
            f.write(json.dumps(query_results) + "\n")
    
    # Run GPT scoring
    print("\nRunning GPT scoring...")
    gpt_results = gpt_scorer.score_evaluation_results(results_path)
    gpt_results_path = os.path.join(output_dir, "gpt_results.json")
    with open(gpt_results_path, 'w') as f:
        json.dump(gpt_results, f, indent=2)
    
    # Run statistical analysis
    print("\nRunning statistical analysis...")
    analysis_results = analyzer.analyze_results(results, gpt_results)
    analysis_path = os.path.join(output_dir, "statistical_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(results, gpt_results, analysis_results, output_dir)
    report_path = os.path.join(output_dir, "evaluation_summary.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")
    print("\nSummary of files generated:")
    print(f"- Raw results: {results_path}")
    print(f"- Per-query results: {per_query_path}")
    print(f"- GPT scores: {gpt_results_path}")
    print(f"- Statistical analysis: {analysis_path}")
    print(f"- Summary report: {report_path}")

def generate_summary_report(
    results: dict,
    gpt_results: dict,
    analysis_results: dict,
    output_dir: str
) -> str:
    """Generate a markdown summary report of the evaluation results."""
    report = ["# Evaluation Summary Report\n"]
    
    # Overall metrics
    report.append("## Overall Metrics\n")
    for method in ["graph_rag", "semantic"]:
        report.append(f"\n### {method.upper()}\n")
        for query_type in ["relationship", "attribute"]:
            metrics = results["results"][method][f"{query_type}_metrics"]
            report.append(f"\n#### {query_type.title()} Queries\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            for metric, value in metrics.items():
                report.append(f"| {metric} | {value:.3f} |")
    
    # Statistical significance
    report.append("\n## Statistical Analysis\n")
    
    # Paired tests
    report.append("\n### Paired Tests\n")
    for test_type in ["t_test", "wilcoxon"]:
        report.append(f"\n#### {test_type.upper()}\n")
        for query_type in ["relationship", "attribute"]:
            report.append(f"\n##### {query_type.title()} Queries\n")
            report.append("| Metric | Statistic | p-value | Effect Size | Significant |")
            report.append("|--------|-----------|---------|-------------|-------------|")
            for metric, test_results in analysis_results["paired_tests"][test_type][query_type].items():
                report.append(
                    f"| {metric} | {test_results['statistic']:.3f} | "
                    f"{test_results['p_value']:.3e} | {test_results['effect_size']:.3f} | "
                    f"{'Yes' if test_results['significant'] else 'No'} |"
                )
    
    # ANOVA results
    report.append("\n### Two-way ANOVA\n")
    for metric, anova_results in analysis_results["anova"].items():
        report.append(f"\n#### {metric.upper()}\n")
        report.append("| Effect | F-statistic | p-value | Effect Size |")
        report.append("|--------|-------------|---------|-------------|")
        for effect, stats in anova_results.items():
            report.append(
                f"| {effect} | {stats['f_statistic']:.3f} | "
                f"{stats['p_value']:.3e} | {stats['effect_size']:.3f} |"
            )
    
    # GPT scoring
    report.append("\n## GPT Scoring Results\n")
    for method in ["graph_rag", "semantic"]:
        report.append(f"\n### {method.upper()}\n")
        for query_type in ["relationship", "attribute"]:
            scores = gpt_results["gpt_scores"][method][f"{query_type}_aggregate"]
            report.append(f"\n#### {query_type.title()} Queries\n")
            report.append("| Criterion | Score |")
            report.append("|-----------|-------|")
            for criterion, score in scores.items():
                report.append(f"| {criterion} | {score:.3f} |")
    
    # Clustering analysis
    report.append("\n## Clustering Analysis\n")
    clustering = analysis_results["clustering_and_pca"]["clustering"]
    report.append(f"\nSilhouette Score: {clustering['silhouette_score']:.3f}\n")
    
    for cluster_id, chars in clustering["cluster_characteristics"].items():
        report.append(f"\n### {cluster_id}\n")
        report.append(f"Size: {chars['size']}\n")
        report.append("Method Distribution:\n")
        for method, count in chars["method_distribution"].items():
            report.append(f"- {method}: {count}\n")
        report.append("Query Type Distribution:\n")
        for qtype, count in chars["query_type_distribution"].items():
            report.append(f"- {qtype}: {count}\n")
    
    # Save report
    report_path = Path(output_dir) / "evaluation_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    return "\n".join(report)

def main():
    run_evaluation()

if __name__ == "__main__":
    main() 