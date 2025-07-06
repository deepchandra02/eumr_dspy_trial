import chromadb
import uuid
from typing import List, Dict


class VectorDatabase:
    """Handles ChromaDB vector database operations"""
    
    def __init__(self, collection_name: str = "vacuum_reviews", reset_collection: bool = True):
        """Initialize ChromaDB client and collection"""
        self.collection_name = collection_name
        self.chroma_client = chromadb.Client()
        
        # Delete collection if exists (only if reset_collection is True)
        if reset_collection:
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except:
                pass
        
        # Try to get existing collection first, create if doesn't exist
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Vacuum reviews with Gemini embeddings for classification"
                },
            )
    
    def store_embeddings(self, data: List[Dict], embeddings: List[List[float]]):
        """Store embeddings and metadata in ChromaDB"""
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        valid_embeddings = []
        
        for i, (item, embedding) in enumerate(zip(data, embeddings)):
            if embedding is not None:  # Skip failed embeddings
                documents.append(item["text"])
                metadatas.append(
                    {
                        "tags": ",".join(item["tags"]),
                        "dataset_type": item["dataset_type"],
                        "review_id": str(i),
                    }
                )
                ids.append(str(uuid.uuid4()))
                valid_embeddings.append(embedding)
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=valid_embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        print(f"💾 Stored {len(valid_embeddings)} embeddings in ChromaDB")
    
    def search_similar_documents(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """Search for similar documents using embedding"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        
        return results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "status": "ready"
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "count": 0,
                "status": f"error: {e}"
            }