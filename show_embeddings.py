"""
Demonstrate embeddings in our vacuum review system
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def show_embeddings_demo():
    """Show exactly how embeddings work in our system"""
    
    print("🤖 EMBEDDING DEMONSTRATION")
    print("=" * 50)
    
    # Same model we use in the system
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Our vacuum tags
    tags = [
        "#PerformanceAndFunctionality",
        "#MaintenanceAndCleaning", 
        "#DesignAndUsabilityIssues"
    ]
    
    # Example queries our system might infer
    queries = [
        "easy to empty",           # Should match MaintenanceAndCleaning
        "suction power weak",      # Should match PerformanceAndFunctionality  
        "cord is broken"           # Should match DesignAndUsabilityIssues
    ]
    
    print("\n🔧 STEP 1: Creating embeddings...")
    tag_embeddings = model.encode(tags)
    query_embeddings = model.encode(queries)
    
    print(f"✓ Each tag becomes {tag_embeddings.shape[1]} numbers")
    print(f"✓ Each query becomes {query_embeddings.shape[1]} numbers")
    
    print("\n🔧 STEP 2: Calculating similarities...")
    print("(1.0 = perfect match, 0.0 = no match)\n")
    
    # Show similarity matrix
    for i, query in enumerate(queries):
        print(f"Query: '{query}'")
        best_score = 0
        best_tag = ""
        
        for j, tag in enumerate(tags):
            similarity = cosine_similarity(
                [query_embeddings[i]], 
                [tag_embeddings[j]]
            )[0][0]
            
            print(f"  {tag}: {similarity:.3f}")
            
            if similarity > best_score:
                best_score = similarity
                best_tag = tag
        
        print(f"  🎯 Best match: {best_tag} ({best_score:.3f})")
        print()

def show_what_embedding_looks_like():
    """Show what an actual embedding vector looks like"""
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    text = "#PerformanceAndFunctionality"
    embedding = model.encode([text])[0]
    
    print("🔍 What does an embedding actually look like?")
    print(f"Text: '{text}'")
    print(f"Embedding (first 10 numbers): {embedding[:10]}")
    print(f"Full length: {len(embedding)} numbers")
    print("\nThis is how the computer 'understands' the meaning!")

if __name__ == "__main__":
    show_embeddings_demo()
    print("\n" + "="*50)
    show_what_embedding_looks_like()
