import json
import os
from typing import List, Dict


class DataLoader:
    """Handles loading and preprocessing of datasets"""
    
    def __init__(self, data_dir: str = "src/data"):
        """Initialize data loader with data directory"""
        self.data_dir = data_dir
    
    def load_datasets(self) -> List[Dict]:
        """Load all available JSON datasets"""
        datasets = []
        files = [
            "dataset-dev.json",
            "dataset-test.json",
        ]
        
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Add dataset type for tracking
                    dataset_type = filename.replace(".json", "")
                    for item in data:
                        item["dataset_type"] = dataset_type
                    datasets.extend(data)
                    print(f"📄 Loaded {len(data)} items from {filename}")
        
        return datasets
    
    def load_single_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a specific dataset by name"""
        filename = f"{dataset_name}.json"
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Add dataset type for tracking
            for item in data:
                item["dataset_type"] = dataset_name
        
        print(f"📄 Loaded {len(data)} items from {filename}")
        return data
    
    def load_dev_dataset(self) -> List[Dict]:
        """Load only dev dataset"""
        return self.load_single_dataset("dataset-dev")
    
    def load_test_sample(self, n_samples: int = 10) -> List[Dict]:
        """Load first n samples from test dataset"""
        test_data = self.load_single_dataset("dataset-test")
        return test_data[:n_samples]
    
    def get_all_unique_tags(self, data: List[Dict]) -> set:
        """Extract all unique tags from dataset"""
        all_tags = set()
        for item in data:
            if "tags" in item:
                all_tags.update(item["tags"])
        return all_tags
    
    def get_dataset_stats(self, data: List[Dict]) -> Dict:
        """Get statistics about the dataset"""
        if not data:
            return {"total_items": 0, "unique_tags": 0, "avg_text_length": 0}
        
        total_items = len(data)
        unique_tags = len(self.get_all_unique_tags(data))
        avg_text_length = sum(len(item.get("text", "")) for item in data) / total_items
        
        # Count items by dataset type
        dataset_counts = {}
        for item in data:
            dataset_type = item.get("dataset_type", "unknown")
            dataset_counts[dataset_type] = dataset_counts.get(dataset_type, 0) + 1
        
        return {
            "total_items": total_items,
            "unique_tags": unique_tags,
            "avg_text_length": round(avg_text_length, 2),
            "dataset_breakdown": dataset_counts
        }