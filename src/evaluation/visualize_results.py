import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np

class ResultsVisualizer:
    def __init__(self, results_path: str):
        """Initialize with path to evaluation results JSON file."""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
    def plot_metric_comparison(self, output_dir: str):
        """Create comparison plots for all metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        for method in ["graph_rag", "semantic"]:
            for query_type in ["relationship", "attribute"]:
                metrics = self.results["results"][method][query_type]
                for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                    plot_data.append({
                        "Method": method.upper(),
                        "Query Type": query_type.title(),
                        "Metric": metric.upper(),
                        "Value": metrics[f"mean_{metric}"],
                        "Std": metrics[f"std_{metric}"]
                    })
        
        df = pd.DataFrame(plot_data)
        
        # Create comparison plots
        for metric in df["Metric"].unique():
            plt.figure(figsize=(10, 6))
            metric_data = df[df["Metric"] == metric]
            
            # Create grouped bar plot
            ax = sns.barplot(
                data=metric_data,
                x="Query Type",
                y="Value",
                hue="Method",
                palette="Set2"
            )
            
            # Add error bars
            for i, bar in enumerate(ax.patches):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                std = metric_data.iloc[i]["Std"]
                plt.errorbar(x, y, yerr=std, fmt="none", color="black", capsize=5)
            
            plt.title(f"{metric} Comparison")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{metric.lower()}_comparison.png"))
            plt.close()
            
    def plot_improvement_heatmap(self, output_dir: str):
        """Create heatmap showing improvement percentages."""
        # Calculate improvements
        improvements = []
        for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
            for query_type in ["relationship", "attribute"]:
                graph_rag = self.results["results"]["graph_rag"][query_type]
                semantic = self.results["results"]["semantic"][query_type]
                
                improvement = (
                    (graph_rag[f"mean_{metric}"] - semantic[f"mean_{metric}"]) /
                    semantic[f"mean_{metric}"] * 100
                )
                
                improvements.append({
                    "Metric": metric.upper(),
                    "Query Type": query_type.title(),
                    "Improvement": improvement
                })
        
        # Create heatmap
        df = pd.DataFrame(improvements)
        pivot_df = df.pivot(index="Metric", columns="Query Type", values="Improvement")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Improvement (%)"}
        )
        
        plt.title("GraphRAG vs Semantic Retrieval Improvement")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "improvement_heatmap.png"))
        plt.close()
        
    def plot_metric_distribution(self, output_dir: str):
        """Create violin plots showing metric distributions."""
        # Prepare data
        plot_data = []
        for method in ["graph_rag", "semantic"]:
            for query_type in ["relationship", "attribute"]:
                metrics = self.results["results"][method][query_type]
                for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                    # Generate synthetic data based on mean and std
                    values = np.random.normal(
                        metrics[f"mean_{metric}"],
                        metrics[f"std_{metric}"],
                        1000
                    )
                    values = np.clip(values, 0, 1)  # Clip to valid range
                    
                    for value in values:
                        plot_data.append({
                            "Method": method.upper(),
                            "Query Type": query_type.title(),
                            "Metric": metric.upper(),
                            "Value": value
                        })
        
        df = pd.DataFrame(plot_data)
        
        # Create violin plots
        for metric in df["Metric"].unique():
            plt.figure(figsize=(12, 6))
            metric_data = df[df["Metric"] == metric]
            
            sns.violinplot(
                data=metric_data,
                x="Query Type",
                y="Value",
                hue="Method",
                split=True,
                palette="Set2"
            )
            
            plt.title(f"{metric} Distribution")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{metric.lower()}_distribution.png"))
            plt.close()
            
    def generate_report(self, output_dir: str):
        """Generate comprehensive evaluation report with all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all plots
        self.plot_metric_comparison(output_dir)
        self.plot_improvement_heatmap(output_dir)
        self.plot_metric_distribution(output_dir)
        
        # Generate summary statistics
        summary = {
            "timestamp": self.results["timestamp"],
            "overall_improvement": {},
            "best_performing_metrics": {},
            "worst_performing_metrics": {}
        }
        
        for query_type in ["relationship", "attribute"]:
            improvements = []
            for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                graph_rag = self.results["results"]["graph_rag"][query_type]
                semantic = self.results["results"]["semantic"][query_type]
                
                improvement = (
                    (graph_rag[f"mean_{metric}"] - semantic[f"mean_{metric}"]) /
                    semantic[f"mean_{metric}"] * 100
                )
                improvements.append((metric, improvement))
            
            improvements.sort(key=lambda x: x[1], reverse=True)
            summary["overall_improvement"][query_type] = np.mean([imp for _, imp in improvements])
            summary["best_performing_metrics"][query_type] = improvements[0]
            summary["worst_performing_metrics"][query_type] = improvements[-1]
        
        # Save summary
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Generated evaluation report in {output_dir}")
        print("\nSummary:")
        for query_type in ["relationship", "attribute"]:
            print(f"\n{query_type.title()} Queries:")
            print(f"Overall improvement: {summary['overall_improvement'][query_type]:+.1f}%")
            print(f"Best metric: {summary['best_performing_metrics'][query_type][0]} ({summary['best_performing_metrics'][query_type][1]:+.1f}%)")
            print(f"Worst metric: {summary['worst_performing_metrics'][query_type][0]} ({summary['worst_performing_metrics'][query_type][1]:+.1f}%)")

def main():
    # Example usage
    results_path = "results/evaluation/evaluation_results_20240321_123456.json"  # Update with actual path
    output_dir = "results/visualization"
    
    visualizer = ResultsVisualizer(results_path)
    visualizer.generate_report(output_dir)

if __name__ == "__main__":
    main() 