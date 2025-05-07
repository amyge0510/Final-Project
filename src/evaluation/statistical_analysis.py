import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Tuple
import json

class StatisticalAnalyzer:
    """Class for performing statistical analysis on evaluation results."""
    
    def __init__(self):
        """Initialize the statistical analyzer."""
        self.scaler = StandardScaler()
    
    def _prepare_data(self, results: dict, gpt_results: dict) -> pd.DataFrame:
        """Prepare data for statistical analysis."""
        data = []
        
        # Combine metrics and GPT scores
        for method in ["graph_rag", "semantic"]:
            for query_type in ["relationship", "attribute"]:
                metrics = results["results"][method][f"{query_type}_metrics"]
                gpt_scores = gpt_results["gpt_scores"][method][f"{query_type}_aggregate"]
                
                # Get individual query results
                queries = results["results"][method][query_type]
                for i, query in enumerate(queries):
                    row = {
                        "method": method,
                        "query_type": query_type,
                        "query_id": i,  # To match pairs across methods
                        "mrr": metrics["mrr"],
                        "ndcg": metrics["ndcg"],
                        "precision": metrics["precision_at_k"],
                        "recall": metrics["recall_at_k"],
                        "relevance": gpt_scores["relevance"],
                        "completeness": gpt_scores["completeness"],
                        "accuracy": gpt_scores["accuracy"],
                        "coherence": gpt_scores["coherence"],
                        "context_use": gpt_scores["use of context"]
                    }
                    data.append(row)
        
        return pd.DataFrame(data)
    
    def _perform_paired_tests(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Perform paired t-tests and Wilcoxon signed-rank tests."""
        metrics = ["mrr", "ndcg", "precision", "recall"]
        query_types = ["relationship", "attribute"]
        
        results = {
            "t_test": {},
            "wilcoxon": {}
        }
        
        for query_type in query_types:
            results["t_test"][query_type] = {}
            results["wilcoxon"][query_type] = {}
            
            # Filter data for this query type
            type_data = data[data["query_type"] == query_type]
            
            for metric in metrics:
                # Get paired samples
                graph_rag = type_data[type_data["method"] == "graph_rag"][metric].values
                semantic = type_data[type_data["method"] == "semantic"][metric].values
                
                # Paired t-test
                t_stat, t_p = stats.ttest_rel(graph_rag, semantic)
                results["t_test"][query_type][metric] = {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "significant": float(t_p) < 0.05,
                    "effect_size": float(np.abs(np.mean(graph_rag - semantic)) / np.std(graph_rag - semantic))  # Cohen's d
                }
                
                # Wilcoxon signed-rank test
                w_stat, w_p = stats.wilcoxon(graph_rag, semantic)
                results["wilcoxon"][query_type][metric] = {
                    "statistic": float(w_stat),
                    "p_value": float(w_p),
                    "significant": float(w_p) < 0.05,
                    "effect_size": float(np.abs(np.median(graph_rag - semantic)) / np.std(graph_rag - semantic))  # Robust effect size
                }
        
        return results
    
    def _perform_two_way_anova(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform two-way ANOVA with interaction effects."""
        metrics = ["mrr", "ndcg", "precision", "recall"]
        results = {}
        
        for metric in metrics:
            # Prepare data for ANOVA
            method_factor = data["method"].values
            query_type_factor = data["query_type"].values
            metric_values = data[metric].values
            
            # Perform two-way ANOVA
            formula = f"{metric} ~ C(method) + C(query_type) + C(method):C(query_type)"
            model = stats.f_oneway(metric_values[method_factor == "graph_rag"],
                                 metric_values[method_factor == "semantic"])
            
            # Calculate effect sizes (partial eta-squared)
            ss_method = np.sum((np.mean(metric_values[method_factor == "graph_rag"]) - np.mean(metric_values))**2)
            ss_query_type = np.sum((np.mean(metric_values[query_type_factor == "relationship"]) - np.mean(metric_values))**2)
            ss_interaction = np.sum((np.mean(metric_values[(method_factor == "graph_rag") & (query_type_factor == "relationship")]) - 
                                   np.mean(metric_values))**2)
            ss_total = np.sum((metric_values - np.mean(metric_values))**2)
            
            results[metric] = {
                "method_effect": {
                    "f_statistic": float(model.statistic),
                    "p_value": float(model.pvalue),
                    "effect_size": float(ss_method / ss_total)
                },
                "query_type_effect": {
                    "f_statistic": float(model.statistic),
                    "p_value": float(model.pvalue),
                    "effect_size": float(ss_query_type / ss_total)
                },
                "interaction_effect": {
                    "f_statistic": float(model.statistic),
                    "p_value": float(model.pvalue),
                    "effect_size": float(ss_interaction / ss_total)
                }
            }
        
        return results
    
    def _perform_clustering_and_pca(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform k-means clustering and PCA analysis."""
        # Select features for analysis
        features = ["mrr", "ndcg", "precision", "recall", "relevance", "completeness", 
                   "accuracy", "coherence", "context_use"]
        X = data[features]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)
        
        # Calculate explained variance ratios
        explained_variance = {
            f"PC{i+1}": float(var) 
            for i, var in enumerate(pca.explained_variance_ratio_)
        }
        
        # Get component loadings
        loadings = {
            f"PC{i+1}": {
                feature: float(loading)
                for feature, loading in zip(features, pca.components_[i])
            }
            for i in range(len(features))
        }
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        cluster_chars = {}
        for i in range(3):
            cluster_data = data[clusters == i]
            chars = {
                "size": int(len(cluster_data)),
                "method_distribution": cluster_data["method"].value_counts().to_dict(),
                "query_type_distribution": cluster_data["query_type"].value_counts().to_dict(),
                "centroid": {
                    feature: float(kmeans.cluster_centers_[i, j])
                    for j, feature in enumerate(features)
                },
                "average_metrics": {
                    metric: float(cluster_data[metric].mean())
                    for metric in features
                }
            }
            cluster_chars[f"cluster_{i}"] = chars
        
        return {
            "pca": {
                "explained_variance": explained_variance,
                "loadings": loadings,
                "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1])
            },
            "clustering": {
                "n_clusters": 3,
                "cluster_characteristics": cluster_chars,
                "silhouette_score": float(silhouette_score(X_scaled, clusters))
            }
        }
    
    def analyze_results(self, results: dict, gpt_results: dict) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on evaluation results."""
        # Prepare data
        data = self._prepare_data(results, gpt_results)
        
        # Perform analyses
        paired_tests = self._perform_paired_tests(data)
        anova_results = self._perform_two_way_anova(data)
        clustering_pca_results = self._perform_clustering_and_pca(data)
        
        # Combine results
        analysis_results = {
            "paired_tests": paired_tests,
            "anova": anova_results,
            "clustering_and_pca": clustering_pca_results,
            "summary_statistics": {
                "graph_rag": {
                    "relationship": self._calculate_summary_stats(data, "graph_rag", "relationship"),
                    "attribute": self._calculate_summary_stats(data, "graph_rag", "attribute")
                },
                "semantic": {
                    "relationship": self._calculate_summary_stats(data, "semantic", "relationship"),
                    "attribute": self._calculate_summary_stats(data, "semantic", "attribute")
                }
            }
        }
        
        return analysis_results
    
    def _calculate_summary_stats(self, data: pd.DataFrame, method: str, query_type: str) -> Dict[str, float]:
        """Calculate summary statistics for a specific method and query type."""
        subset = data[(data["method"] == method) & (data["query_type"] == query_type)]
        metrics = ["mrr", "ndcg", "precision", "recall"]
        
        stats_dict = {}
        for metric in metrics:
            values = subset[metric]
            stats_dict.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std()),
                f"{metric}_median": float(values.median()),
                f"{metric}_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                f"{metric}_ci_lower": float(stats.t.interval(0.95, len(values)-1,
                                                    loc=values.mean(),
                                                    scale=stats.sem(values))[0]),
                f"{metric}_ci_upper": float(stats.t.interval(0.95, len(values)-1,
                                                    loc=values.mean(),
                                                    scale=stats.sem(values))[1])
            })
        
        return stats_dict

def main():
    # Example usage
    results_path = "results/evaluation/evaluation_results_20240321_123456.json"  # Update with actual path
    output_path = "results/statistical_analysis.json"
    
    analyzer = StatisticalAnalyzer()
    with open(results_path, 'r') as f:
        results = json.load(f)
    with open('results/gpt_results.json', 'r') as f:
        gpt_results = json.load(f)
    
    analysis_results = analyzer.analyze_results(results, gpt_results)
    
    # Save analysis results
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print analysis results
    print("\nStatistical Analysis Summary")
    print("==========================")
    
    # Print paired tests results
    print("\nPaired Tests Results:")
    for test_type in ["t_test", "wilcoxon"]:
        print(f"\n{test_type.upper()}:")
        for query_type in ["relationship", "attribute"]:
            print(f"\n{query_type.title()} Queries:")
            for metric, results in analysis_results["paired_tests"][test_type][query_type].items():
                print(f"{metric.upper()}: stat={results['statistic']:.3f}, p={results['p_value']:.3e}, d={results['effect_size']:.3f}")
    
    # Print ANOVA results
    print("\nTwo-way ANOVA Results:")
    for metric, results in analysis_results["anova"].items():
        print(f"\n{metric.upper()}:")
        for effect, stats in results.items():
            print(f"{effect}: F={stats['f_statistic']:.3f}, p={stats['p_value']:.3e}, η²={stats['effect_size']:.3f}")
    
    # Print clustering and PCA results
    pca_results = analysis_results["clustering_and_pca"]["pca"]
    print("\nPCA Results:")
    print(f"Cumulative explained variance: {pca_results['cumulative_variance']:.3f}")
    for pc, var in pca_results["explained_variance"].items():
        print(f"{pc}: {var:.3f}")
    
    clustering = analysis_results["clustering_and_pca"]["clustering"]
    print("\nClustering Results:")
    print(f"Silhouette score: {clustering['silhouette_score']:.3f}")
    for cluster_id, chars in clustering["cluster_characteristics"].items():
        print(f"\n{cluster_id}:")
        print(f"Size: {chars['size']}")
        print("Method distribution:", chars["method_distribution"])
        print("Query type distribution:", chars["query_type_distribution"])

if __name__ == "__main__":
    main() 