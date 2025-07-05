"""
Data cleaning script for EUMR DSPy Trial

This script processes JSON data files to:
1. Rename "expected" key to "tags"
2. Remove "sentiment" key-value pairs

Usage: python data_clean.py
"""

import json
import os
from pathlib import Path


def clean_data_file(file_path):
    """
    Clean a single JSON data file by renaming 'expected' to 'tags' and removing 'sentiment'
    
    Args:
        file_path (str): Path to the JSON file to clean
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each item in the data
        for item in data:
            # Rename 'expected' to 'tags'
            if 'expected' in item:
                item['tags'] = item.pop('expected')
            
            # Remove 'sentiment' key if it exists
            if 'sentiment' in item:
                del item['sentiment']
        
        # Write the cleaned data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False)
        
        print(f"✓ Cleaned: {file_path}")
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")


def main():
    """Main function to clean all JSON data files"""
    # Get the script's directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "src" / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find all JSON files in the data directory
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in the data directory")
        return
    
    print(f"Found {len(json_files)} JSON files to clean:")
    for file in json_files:
        print(f"  - {file.name}")
    
    print("\nCleaning files...")
    
    # Clean each JSON file
    for json_file in json_files:
        clean_data_file(json_file)
    
    print(f"\nData cleaning complete! Processed {len(json_files)} files.")


if __name__ == "__main__":
    main()
