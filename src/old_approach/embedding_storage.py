import json
import os
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class EmbeddingStorage:
    """Handles persistent storage and retrieval of embeddings"""
    
    def __init__(self, storage_dir: str = "src/embeddings"):
        """Initialize embedding storage with specified directory"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.embeddings_file = self.storage_dir / "embeddings.pkl"
        self.metadata_file = self.storage_dir / "metadata.json"
        self.hash_file = self.storage_dir / "data_hash.txt"
    
    def _calculate_data_hash(self, data: List[Dict]) -> str:
        """Calculate hash of input data to detect changes"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _load_stored_hash(self) -> Optional[str]:
        """Load previously stored data hash"""
        if self.hash_file.exists():
            with open(self.hash_file, 'r') as f:
                return f.read().strip()
        return None
    
    def _save_hash(self, data_hash: str):
        """Save data hash to file"""
        with open(self.hash_file, 'w') as f:
            f.write(data_hash)
    
    def embeddings_exist(self, data: List[Dict]) -> bool:
        """Check if embeddings exist for the given data"""
        if not all([self.embeddings_file.exists(), self.metadata_file.exists(), self.hash_file.exists()]):
            return False
        
        # Check if data has changed
        current_hash = self._calculate_data_hash(data)
        stored_hash = self._load_stored_hash()
        
        return current_hash == stored_hash
    
    def save_embeddings(self, data: List[Dict], embeddings: List[List[float]]):
        """Save embeddings and metadata to disk"""
        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata
        metadata = {
            'count': len(embeddings),
            'data_sample': data[:5] if len(data) > 5 else data,  # Store first 5 items as sample
            'embedding_dim': len(embeddings[0]) if embeddings else 0
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data hash
        data_hash = self._calculate_data_hash(data)
        self._save_hash(data_hash)
        
        print(f"💾 Saved {len(embeddings)} embeddings to {self.storage_dir}")
    
    def load_embeddings(self) -> Tuple[List[List[float]], Dict]:
        """Load embeddings and metadata from disk"""
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"No embeddings found at {self.embeddings_file}")
        
        # Load embeddings
        with open(self.embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        print(f"📂 Loaded {len(embeddings)} embeddings from {self.storage_dir}")
        return embeddings, metadata
    
    def get_storage_info(self) -> Dict:
        """Get information about stored embeddings"""
        if not self.embeddings_file.exists():
            return {"exists": False}
        
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        file_size = self.embeddings_file.stat().st_size / (1024 * 1024)  # Size in MB
        
        return {
            "exists": True,
            "file_size_mb": round(file_size, 2),
            "metadata": metadata,
            "storage_path": str(self.storage_dir)
        }
    
    def clear_storage(self):
        """Clear all stored embeddings and metadata"""
        for file_path in [self.embeddings_file, self.metadata_file, self.hash_file]:
            if file_path.exists():
                file_path.unlink()
        print(f"🗑️  Cleared embedding storage at {self.storage_dir}")