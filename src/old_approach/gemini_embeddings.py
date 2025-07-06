from typing import List
from tqdm import tqdm
from google import genai


class GeminiEmbeddings:
    """Handles Gemini API embedding generation"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client"""
        self.client = genai.Client(api_key=api_key)
    
    def create_embeddings_batch(
        self, texts: List[str], task_type: str = "classification"
    ) -> List[List[float]]:
        """Create embeddings using Gemini API with batch processing"""
        embeddings = []
        
        # Process in batches (Gemini supports batch embedding)
        batch_size = 100  # Adjust based on API limits
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i : i + batch_size]
            
            try:
                result = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch_texts,
                    config={"task_type": task_type},
                )
                
                # Extract embedding values
                batch_embeddings = [emb.values for emb in result.embeddings]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add placeholder embeddings for failed batch
                embeddings.extend([None] * len(batch_texts))
        
        return embeddings
    
    def create_single_embedding(self, text: str, task_type: str = "classification") -> List[float]:
        """Create embedding for a single text"""
        try:
            result = self.client.models.embed_content(
                model="text-embedding-004",
                contents=[text],
                config={"task_type": task_type},
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error creating embedding for text: {e}")
            return None