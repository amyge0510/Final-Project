import os
import json
import yaml
from typing import Dict, List, Any, Set
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class AmazonMetaParser:
    def __init__(self, file_path: str, config_path: str = "config/data_config.yaml"):
        self.file_path = file_path
        self.config = self._load_config(config_path)
        self.products = []
        self.users = set()
        self.reviews = []
        self.categories = set()
        self.co_purchases = {}  # Track co-purchase relationships
        self.review_relationships = {}  # Track review-based relationships
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def parse(self) -> Dict[str, Any]:
        """Parse the Amazon metadata file and return structured data."""
        current_product = None
        current_reviews = []
        product_count = 0
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            lines = lines[:total_lines // 2]  # Only process the first half
            
            for i, line in enumerate(tqdm(lines, desc="Parsing Amazon metadata")):
                try:
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                        
                    if line.startswith('Id:'):
                        if current_product:
                            self._save_product(current_product, current_reviews)
                            product_count += 1
                            if product_count >= self.config['max_products']:
                                break
                        current_product = {'id': line.split(':')[1].strip()}
                        current_reviews = []
                        
                    elif line.startswith('ASIN:'):
                        if current_product:
                            current_product['asin'] = line.split(':')[1].strip()
                        
                    elif line.startswith('title:'):
                        if current_product:
                            current_product['title'] = line.split(':', 1)[1].strip()
                        
                    elif line.startswith('group:'):
                        if current_product:
                            current_product['group'] = line.split(':')[1].strip()
                            # Only process books
                            if current_product['group'] != 'Book':
                                current_product = None
                                continue
                        
                    elif line.startswith('salesrank:'):
                        if current_product:
                            try:
                                current_product['salesrank'] = int(line.split(':')[1].strip())
                            except ValueError:
                                current_product['salesrank'] = 0
                        
                    elif line.startswith('similar:'):
                        if current_product:
                            parts = line.split(':')[1].strip().split()
                            if parts and parts[0] != '0':
                                current_product['similar'] = parts[1:]
                                # Track co-purchase relationships
                                self.co_purchases[current_product['asin']] = parts[1:]
                            
                    elif line.startswith('categories:'):
                        if current_product:
                            current_product['categories'] = []
                        
                    elif line.startswith('|'):
                        if current_product and 'categories' in current_product:
                            try:
                                category_path = line.strip('|').split('|')
                                category_parts = category_path[-1].split('[')
                                if len(category_parts) > 1:
                                    category_id = category_parts[-1].strip(']')
                                    current_product['categories'].append({
                                        'path': category_path,
                                        'id': category_id
                                    })
                                    self.categories.add(category_id)
                            except Exception as e:
                                print(f"Error parsing category line {i+1}: {line}")
                        
                    elif line.startswith('reviews:'):
                        if current_product:
                            try:
                                parts = line.split(':')[1].strip().split()
                                if len(parts) >= 6:
                                    current_product['review_stats'] = {
                                        'total': int(parts[1]),
                                        'downloaded': int(parts[3]),
                                        'avg_rating': float(parts[5])
                                    }
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing review stats line {i+1}: {line}")
                        
                    elif line.startswith('20'):  # Review line starting with date
                        if current_product:
                            try:
                                parts = line.split()
                                if len(parts) >= 9:
                                    review = {
                                        'date': datetime.strptime(parts[0], '%Y-%m-%d'),
                                        'customer_id': parts[2],
                                        'rating': int(parts[4]),
                                        'votes': int(parts[6]),
                                        'helpful': int(parts[8])
                                    }
                                    current_reviews.append(review)
                                    self.users.add(review['customer_id'])
                                    
                                    # Track review relationships
                                    if review['customer_id'] not in self.review_relationships:
                                        self.review_relationships[review['customer_id']] = set()
                                    self.review_relationships[review['customer_id']].add(current_product['asin'])
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing review line {i+1}: {line}")
                                
                except Exception as e:
                    print(f"Error processing line {i+1}: {line}")
                    print(f"Error: {str(e)}")
                    continue
                    
        # Save the last product
        if current_product:
            self._save_product(current_product, current_reviews)
            
        return {
            'products': self.products,
            'users': list(self.users),
            'reviews': self.reviews,
            'categories': list(self.categories),
            'co_purchases': self.co_purchases,
            'review_relationships': self.review_relationships
        }
        
    def _save_product(self, product: Dict, reviews: List[Dict]) -> None:
        """Save product and its reviews to the collections."""
        if ('id' in product and 'asin' in product and 
            product.get('group') == 'Book' and  # Only save books
            len(reviews) >= self.config['min_reviews_per_product']):  # Filter by review count
            self.products.append(product)
            for review in reviews:
                review['product_id'] = product['id']
                review['product_asin'] = product['asin']
                self.reviews.append(review)
            
    def save_to_csv(self, output_dir: str) -> None:
        """Save parsed data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save products
        products_df = pd.DataFrame(self.products)
        products_df.to_csv(os.path.join(output_dir, 'products.csv'), index=False)
        
        # Save reviews
        reviews_df = pd.DataFrame(self.reviews)
        reviews_df.to_csv(os.path.join(output_dir, 'reviews.csv'), index=False)
        
        # Save users
        users_df = pd.DataFrame({'user_id': list(self.users)})
        users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
        
        # Save categories
        categories_df = pd.DataFrame({'category_id': list(self.categories)})
        categories_df.to_csv(os.path.join(output_dir, 'categories.csv'), index=False)
        
        # Save co-purchase relationships
        co_purchases_df = pd.DataFrame([
            {'source': source, 'target': target}
            for source, targets in self.co_purchases.items()
            for target in targets
        ])
        co_purchases_df.to_csv(os.path.join(output_dir, 'co_purchases.csv'), index=False)
        
        # Save review relationships
        review_relationships_df = pd.DataFrame([
            {'user_id': user, 'product_asin': asin}
            for user, asins in self.review_relationships.items()
            for asin in asins
        ])
        review_relationships_df.to_csv(os.path.join(output_dir, 'review_relationships.csv'), index=False)

def main():
    # Load configuration
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Parse the data
    parser = AmazonMetaParser(config['raw_data_path'])
    data = parser.parse()
    
    # Save to CSV
    parser.save_to_csv(config['processed_data_path'])
    
    print(f"Processed {len(data['products'])} books")
    print(f"Found {len(data['users'])} unique users")
    print(f"Extracted {len(data['reviews'])} reviews")
    print(f"Identified {len(data['categories'])} categories")
    print(f"Created {len(data['co_purchases'])} co-purchase relationships")
    print(f"Created {len(data['review_relationships'])} review relationships")

if __name__ == '__main__':
    main() 