"""GPT-based scoring for evaluation answers."""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import time

from openai import OpenAI
from tqdm import tqdm

class GPTScorer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT scorer."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=self.api_key)

    def _create_scoring_prompt(self, query: str, answer: str, context: List[Dict[str, Any]]) -> str:
        """Create a prompt for GPT to score an answer."""
        return f"""
Please evaluate the following answer based on the given context and query.

Query: {query}

Context:
{json.dumps(context, indent=2)}

Answer: {answer}

Score each aspect on a scale of 1-5 (1 being lowest, 5 being highest) and provide a brief explanation:

Relevance: [Score] - [Explanation]
Completeness: [Score] - [Explanation]
Accuracy: [Score] - [Explanation]
Coherence: [Score] - [Explanation]
Use of Context: [Score] - [Explanation]
"""

    def score_answer(self, query: str, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score a single answer using GPT."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                prompt = self._create_scoring_prompt(query, answer, context)
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of question-answering systems."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                raw_response = response.choices[0].message.content
                parsed = self._parse_gpt_response(raw_response)
                
                return {
                    'scores': parsed['scores'],
                    'explanations': parsed['explanations'],
                    'raw_response': raw_response
                }
                
            except Exception as e:
                error_str = str(e)
                if "insufficient_quota" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print("Max retries reached for rate limit")
                
                print(f"Error scoring answer: {error_str}")
                return {
                    'error': error_str,
                    'scores': {},
                    'explanations': {}
                }

    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT's scoring response."""
        scores = {}
        explanations = {}
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' not in line:
                    continue
                    
                category, content = line.split(':', 1)
                category = category.strip().lower().replace(' ', '_')  # Normalize field names
                
                if '-' not in content:
                    continue
                    
                score_part, explanation = content.split('-', 1)
                try:
                    score = int(score_part.strip())
                    if 1 <= score <= 5:
                        scores[category] = score
                        explanations[category] = explanation.strip()
                except ValueError:
                    continue
                    
        except Exception as e:
            return {
                'error': str(e),
                'scores': {},
                'explanations': {}
            }
            
        return {
            'scores': scores,
            'explanations': explanations
        }

    def _aggregate_scores(self, scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate scores across multiple answers."""
        if not scores_list:
            return {}
            
        aggregated = {}
        for metric in ['relevance', 'completeness', 'accuracy', 'coherence', 'use_of_context']:
            values = [s.get(metric, 0) for s in scores_list if metric in s]
            if values:
                aggregated[metric] = sum(values) / len(values)
                
        return aggregated

    def score_evaluation_results(self, results_path: str, output_path: str) -> Dict[str, Any]:
        """Score all answers in evaluation results."""
        with open(results_path, 'r') as f:
            results = json.load(f)

        gpt_scores = {
            'graph_rag': {
                'relationship_scores': [],
                'attribute_scores': [],
            },
            'semantic': {
                'relationship_scores': [],
                'attribute_scores': [],
            }
        }

        # Score GraphRAG results
        print("\nScoring graph_rag relationship queries:")
        for query in tqdm(results['results']['graph_rag']['relationship']):
            score = self.score_answer(query['query'], query['answer'], query['context'])
            if 'scores' in score:
                gpt_scores['graph_rag']['relationship_scores'].append(score['scores'])

        print("\nScoring graph_rag attribute queries:")
        for query in tqdm(results['results']['graph_rag']['attribute']):
            score = self.score_answer(query['query'], query['answer'], query['context'])
            if 'scores' in score:
                gpt_scores['graph_rag']['attribute_scores'].append(score['scores'])

        # Score semantic search results
        print("\nScoring semantic relationship queries:")
        for query in tqdm(results['results']['semantic']['relationship']):
            score = self.score_answer(query['query'], query['answer'], query['context'])
            if 'scores' in score:
                gpt_scores['semantic']['relationship_scores'].append(score['scores'])

        print("\nScoring semantic attribute queries:")
        for query in tqdm(results['results']['semantic']['attribute']):
            score = self.score_answer(query['query'], query['answer'], query['context'])
            if 'scores' in score:
                gpt_scores['semantic']['attribute_scores'].append(score['scores'])

        # Aggregate scores
        final_scores = {
            'graph_rag': {
                'relationship_aggregate': self._aggregate_scores(gpt_scores['graph_rag']['relationship_scores']),
                'attribute_aggregate': self._aggregate_scores(gpt_scores['graph_rag']['attribute_scores']),
                'all_scores': gpt_scores['graph_rag']
            },
            'semantic': {
                'relationship_aggregate': self._aggregate_scores(gpt_scores['semantic']['relationship_scores']),
                'attribute_aggregate': self._aggregate_scores(gpt_scores['semantic']['attribute_scores']),
                'all_scores': gpt_scores['semantic']
            }
        }

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'original_results': results,
            'gpt_scores': final_scores
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Print summary
        print("\nGPT-4 Scoring Summary")
        print("====================\n")
        print("GRAPH_RAG Results:\n")
        print("Relationship Queries:")
        for metric, score in final_scores['graph_rag']['relationship_aggregate'].items():
            print(f"{metric.capitalize()}: {score:.2f}")
        print("\nAttribute Queries:")
        for metric, score in final_scores['graph_rag']['attribute_aggregate'].items():
            print(f"{metric.capitalize()}: {score:.2f}")
        
        print("\nSEMANTIC Results:\n")
        print("Relationship Queries:")
        for metric, score in final_scores['semantic']['relationship_aggregate'].items():
            print(f"{metric.capitalize()}: {score:.2f}")
        print("\nAttribute Queries:")
        for metric, score in final_scores['semantic']['attribute_aggregate'].items():
            print(f"{metric.capitalize()}: {score:.2f}")

        return output

def main():
    # Example usage
    results_path = "results/evaluation/evaluation_results_20240321_123456.json"  # Update with actual path
    output_path = "results/gpt_scores.json"
    
    scorer = GPTScorer()
    scorer.score_evaluation_results(results_path, output_path)

if __name__ == "__main__":
    main() 