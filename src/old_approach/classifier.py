from typing import List, Dict, Optional
from tqdm import tqdm
from google import genai


class DocumentClassifier:
    """Handles document classification using similar examples"""
    
    def __init__(self, client: genai.Client, vector_db, gemini_embeddings):
        """Initialize classifier with dependencies"""
        self.client = client
        self.vector_db = vector_db
        self.gemini_embeddings = gemini_embeddings
    
    def classify_document(
        self, document: str, target_tags: List[str] = None, n_similar: int = 5
    ) -> Dict:
        """Classify document based on similar examples"""
        
        # Create embedding for the document
        query_embedding = self.gemini_embeddings.create_single_embedding(
            document, task_type="classification"
        )
        
        if query_embedding is None:
            return {
                "predicted_tags": [],
                "confidence": 0.0,
                "similar_examples": [],
                "tag_frequencies": {},
                "error": "Failed to create embedding"
            }
        
        # Get similar documents
        results = self.vector_db.search_similar_documents(query_embedding, n_results=n_similar)
        
        # Extract tags from similar documents
        similar_tags = []
        similar_docs = []
        similarities = []
        
        for i, metadata in enumerate(results["metadatas"][0]):
            tags = metadata["tags"].split(",") if metadata["tags"] else []
            similar_tags.extend(tags)
            similar_docs.append(
                {
                    "text": results["documents"][0][i][:200] + "...",
                    "tags": tags,
                    "similarity": 1 - results["distances"][0][i],
                }
            )
            similarities.append(1 - results["distances"][0][i])
        
        # Get unique tags with frequency
        tag_counts = {}
        for tag in similar_tags:
            tag = tag.strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Create prompt for LLM classification
        similar_examples_text = "\n".join(
            [
                f"Example {i+1} (similarity: {doc['similarity']:.3f}): {doc['text']} | Tags: {doc['tags']}"
                for i, doc in enumerate(similar_docs[:3])
            ]
        )
        
        available_tags = list(tag_counts.keys())
        if target_tags:
            available_tags = list(set(available_tags + target_tags))
        
        prompt = f"""Based on similar examples, classify this document with appropriate tags.

Document to classify: {document}

Similar examples from database:
{similar_examples_text}

Available tags (with frequency from similar docs): {dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))}
{f"Target tag options: {target_tags}" if target_tags else ""}

Instructions:
- Return only the most relevant tags as a comma-separated list
- Consider the tags from the most similar examples
- Be concise and accurate
- If document doesn't match any examples well, return fewer tags

Tags:"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt]
            )
            
            predicted_tags = [tag.strip() for tag in response.text.strip().split(",")]
            predicted_tags = [tag for tag in predicted_tags if tag]  # Remove empty tags
            
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback: use most frequent tags from similar documents
            predicted_tags = list(
                dict(
                    sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                ).keys()
            )[:3]
        
        return {
            "predicted_tags": predicted_tags,
            "confidence": max(similarities) if similarities else 0.0,
            "similar_examples": similar_docs,
            "tag_frequencies": tag_counts,
        }
    
    def batch_classify_documents(
        self, documents: List[str], target_tags: List[str] = None
    ) -> List[Dict]:
        """Classify multiple documents"""
        results = []
        for doc in tqdm(documents, desc="Classifying documents"):
            result = self.classify_document(doc, target_tags)
            results.append(result)
        return results