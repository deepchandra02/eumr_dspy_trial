"""
OpenAI-powered version for better performance
"""

import dspy
from vacuum_irera import VacuumIReRa, VacuumIReRaConfig, load_vacuum_data
import time
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced config for OpenAI
class VacuumIReRaConfigOpenAI(VacuumIReRaConfig):
    retriever_model_name: str = "text-embedding-3-small"  # OpenAI embeddings
    use_openai_embeddings: bool = True

# OpenAI Embeddings Retriever
class OpenAIVacuumTagRetriever:
    def __init__(self, config):
        self.config = config
        self.tags = [
            "#DesignAndUsabilityIssues",
            "#PerformanceAndFunctionality", 
            "#BatteryAndPowerIssues",
            "#DurabilityAndMaterialConcerns",
            "#MaintenanceAndCleaning",
            "#CustomerExperienceAndExpectations",
            "#ValueForMoneyAndInvestment",
            "#AssemblyAndSetup"
        ]
        
        # Pre-compute tag embeddings with OpenAI
        self.client = openai.OpenAI()
        self.tag_embeddings = self._compute_tag_embeddings()
    
    def _compute_tag_embeddings(self):
        """Get OpenAI embeddings for all tags"""
        response = self.client.embeddings.create(
            input=self.tags,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    
    def retrieve(self, queries):
        """Retrieve using OpenAI embeddings"""
        if not queries:
            return {tag: 0.0 for tag in self.tags}
        
        # Get embeddings for queries
        response = self.client.embeddings.create(
            input=queries,
            model="text-embedding-3-small"
        )
        query_embeddings = [item.embedding for item in response.data]
        
        # Compute similarities
        scores = {}
        for tag, tag_emb in zip(self.tags, self.tag_embeddings):
            similarities = []
            for query_emb in query_embeddings:
                sim = np.dot(tag_emb, query_emb) / (np.linalg.norm(tag_emb) * np.linalg.norm(query_emb))
                similarities.append(sim)
            scores[tag] = float(max(similarities))
        
        return scores

# Enhanced Infer-Retrieve with OpenAI embeddings
class InferRetrieveOpenAI(dspy.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        from vacuum_irera import Infer
        self.infer = Infer(config)
        self.retriever = OpenAIVacuumTagRetriever(config)
    
    def forward(self, text: str) -> dspy.Prediction:
        # Use LM to predict tag queries
        preds = self.infer(text).predictions
        
        # Retrieve tags using OpenAI embeddings
        scores = self.retriever.retrieve(preds)
        
        # Sort tags by relevance score
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)
        
        return dspy.Prediction(predictions=labels)

# Enhanced IReRa with OpenAI
class VacuumIReRaOpenAI(dspy.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use OpenAI-enhanced modules
        self.infer_retrieve = InferRetrieveOpenAI(config)
        from vacuum_irera import Rank
        self.rank = Rank(config)
        
        self.rank_skip = config.rank_skip
        self.rank_topk = config.rank_topk
    
    def forward(self, text: str) -> dspy.Prediction:
        prediction = self.infer_retrieve(text)
        labels = prediction.predictions
        
        options = labels[:self.rank_topk]
        
        if not self.rank_skip:
            reranked = self.rank(text, options).predictions
            selected_options = [o for o in reranked if o in options]
            selected_options += [o for o in options if o not in selected_options]
        else:
            selected_options = options
        
        return dspy.Prediction(predictions=selected_options)

def setup_openai():
    """Configure DSPy with OpenAI GPT-4"""
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Use GPT-4 for better performance
    lm = dspy.LM('gpt-4o-mini', max_tokens=150)  # Cost-effective option
    # For best performance: lm = dspy.LM('gpt-4', max_tokens=150)
    dspy.configure(lm=lm)
    return True

def run_openai_test():
    if not setup_openai():
        return
    
    print("🚀 Running OpenAI-Enhanced IReRa...")
    
    config = VacuumIReRaConfigOpenAI()
    train_examples = load_vacuum_data(config.dataset_dev_path)
    test_examples = load_vacuum_data(config.dataset_test_path)
    
    # Create enhanced program
    program = VacuumIReRaOpenAI(config)
    
    # Test on more examples without delays
    print("\n📊 Testing with OpenAI embeddings + GPT-4o-mini...")
    
    results = []
    test_size = min(10, len(test_examples))  # Test 10 examples
    
    for i, example in enumerate(test_examples[:test_size]):
        prediction = program(example.text)
        results.append({
            'predicted': prediction.predictions,
            'actual': example.label,
            'text': example.text[:100] + "..."
        })
        print(f"Example {i+1}/{test_size}: ✓")
    
    print(f"\n🎯 Enhanced Results:")
    total_accuracy = 0
    perfect_matches = 0
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"  Text: {result['text']}")
        print(f"  Predicted: {result['predicted'][:3]}")
        print(f"  Actual: {result['actual']}")
        
        predicted_set = set(result['predicted'][:3])
        actual_set = set(result['actual'])
        overlap = len(predicted_set.intersection(actual_set))
        accuracy = overlap/len(actual_set) if len(actual_set) > 0 else 0
        total_accuracy += accuracy
        
        if overlap == len(actual_set):
            perfect_matches += 1
            
        print(f"  Accuracy: {overlap}/{len(actual_set)} = {accuracy:.2f}")
    
    avg_accuracy = total_accuracy/test_size
    print(f"\n🏆 Results Summary:")
    print(f"  Average Accuracy: {avg_accuracy:.2f}")
    print(f"  Perfect Matches: {perfect_matches}/{test_size} ({perfect_matches/test_size*100:.1f}%)")
    print(f"  Total Examples: {test_size}")

if __name__ == "__main__":
    run_openai_test()
