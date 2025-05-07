import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class DataIngestor:
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def process_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw product data into structured format."""
        processed = {
            'asin': product_data.get('asin'),
            'title': product_data.get('title'),
            'price': product_data.get('price'),
            'description': product_data.get('description'),
            'brand': product_data.get('brand'),
            'categories': product_data.get('category', []),
            'features': product_data.get('feature', []),
            'also_bought': product_data.get('related', {}).get('also_bought', []),
            'reviews': product_data.get('reviews', [])
        }
        return processed

    def save_processed_data(self, data: Dict[str, Any], output_path: str):
        """Save processed data to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self, input_file: str, output_file: str):
        """Run the data ingestion pipeline."""
        print(f"Processing data from {input_file}")
        
        # Load and process data
        raw_data = self.load_json_data(input_file)
        processed_data = self.process_product_data(raw_data)
        
        # Save processed data
        self.save_processed_data(processed_data, output_file)
        print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    ingestor = DataIngestor()
    
    # You would typically run this with actual data files
    # ingestor.run(
    #     input_file="data/raw/amazon_products.json",
    #     output_file="data/processed/processed_products.json"
    # ) 