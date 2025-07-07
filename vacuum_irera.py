"""
Vacuum Review Classification using IReRa (Infer-Retrieve-Rank)
Based on the original IReRa method for extreme multi-label classification
"""

import json
import dspy
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

# Configuration
@dataclass
class VacuumIReRaConfig:
    infer_signature_name: str = "infer_vacuum"
    rank_signature_name: str = "rank_vacuum"
    rank_topk: int = 8  # All 8 possible tags
    rank_skip: bool = False
    retriever_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    ontology_path: str = "src/tags.py"
    dataset_dev_path: str = "src/data/raw-data/dataset-dev.json"
    dataset_test_path: str = "src/data/raw-data/dataset-test.json"

# Domain-specific signatures for vacuum reviews
class InferSignatureVacuum(dspy.Signature):
    """Given a vacuum cleaner review, identify all relevant product aspects and issues mentioned."""
    
    text = dspy.InputField(prefix="Review:")
    output = dspy.OutputField(
        prefix="Aspects:",
        desc="list of comma-separated vacuum product aspects and issues",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )

class RankSignatureVacuum(dspy.Signature):
    """Given a vacuum review, select the most relevant tags from the options that best categorize the review content."""
    
    text = dspy.InputField(prefix="Review:")
    options = dspy.InputField(
        prefix="Tag Options:",
        desc="List of comma-separated tag options to choose from",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    output = dspy.OutputField(
        prefix="Tags:",
        desc="list of comma-separated relevant tags",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )

# Supported signatures dictionary
supported_signatures = {
    "infer_vacuum": InferSignatureVacuum,
    "rank_vacuum": RankSignatureVacuum,
}

# Utility functions
def extract_labels_from_strings(text, do_lower: bool = False, strip_punct: bool = False) -> List[str]:
    """Extract labels from comma-separated string output"""
    if not text:
        return []
    
    # Handle both string and list inputs from DSPy
    if isinstance(text, list):
        if len(text) > 0:
            text = str(text[0])  # Take first item and convert to string
        else:
            return []
    
    # Convert to string if it's not already
    text = str(text)
    
    # Split by commas and clean up
    labels = [label.strip() for label in text.split(',')]
    labels = [label for label in labels if label]  # Remove empty strings
    
    if do_lower:
        labels = [label.lower() for label in labels]
    
    if strip_punct:
        import string
        labels = [label.strip(string.punctuation) for label in labels]
    
    return labels

def load_vacuum_tags():
    """Load the 8 vacuum review tags"""
    return [
        "#DesignAndUsabilityIssues",
        "#PerformanceAndFunctionality", 
        "#BatteryAndPowerIssues",
        "#DurabilityAndMaterialConcerns",
        "#MaintenanceAndCleaning",
        "#CustomerExperienceAndExpectations",
        "#ValueForMoneyAndInvestment",
        "#AssemblyAndSetup"
    ]

def load_vacuum_data(file_path: str) -> List[Dict[str, Any]]:
    """Load vacuum review data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DSPy Example format
    examples = []
    for item in data:
        examples.append(dspy.Example(
            text=item['text'],
            label=item['tags']
        ).with_inputs('text'))
    
    return examples

# Simple Retriever for small tag set
class VacuumTagRetriever:
    def __init__(self, config: VacuumIReRaConfig):
        self.config = config
        self.tags = load_vacuum_tags()
        self.model = SentenceTransformer(config.retriever_model_name)
        
        # Pre-compute tag embeddings
        self.tag_embeddings = self.model.encode(self.tags)
    
    def retrieve(self, queries: List[str]) -> Dict[str, float]:
        """Retrieve relevant tags based on queries"""
        if not queries:
            return {tag: 0.0 for tag in self.tags}
        
        # Encode queries
        query_embeddings = self.model.encode(queries)
        
        # Compute similarities
        scores = {}
        for tag, tag_emb in zip(self.tags, self.tag_embeddings):
            # Max similarity across all queries for this tag
            similarities = cosine_similarity([tag_emb], query_embeddings)[0]
            scores[tag] = float(np.max(similarities))
        
        return scores

# Core modules
class Infer(dspy.Module):
    def __init__(self, config: VacuumIReRaConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(supported_signatures[config.infer_signature_name])
    
    def forward(self, text: str) -> dspy.Prediction:
        output = self.cot(text=text).completions.output
        parsed_outputs = set(extract_labels_from_strings(output, do_lower=False, strip_punct=False))
        return dspy.Prediction(predictions=list(parsed_outputs))

class Rank(dspy.Module):
    def __init__(self, config: VacuumIReRaConfig):
        super().__init__()
        self.config = config
        self.cot = dspy.ChainOfThought(supported_signatures[config.rank_signature_name])
    
    def forward(self, text: str, options: List[str]) -> dspy.Prediction:
        output = self.cot(text=text, options=options).completions.output
        parsed_outputs = extract_labels_from_strings(output, do_lower=False, strip_punct=False)
        return dspy.Prediction(predictions=parsed_outputs)

class InferRetrieve(dspy.Module):
    def __init__(self, config: VacuumIReRaConfig):
        super().__init__()
        self.config = config
        self.infer = Infer(config)
        self.retriever = VacuumTagRetriever(config)
    
    def forward(self, text: str) -> dspy.Prediction:
        # Use LM to predict tag queries
        preds = self.infer(text).predictions
        
        # Retrieve tags based on queries
        scores = self.retriever.retrieve(preds)
        
        # Sort tags by relevance score
        labels = sorted(scores, key=lambda k: scores[k], reverse=True)
        
        return dspy.Prediction(predictions=labels)

class VacuumIReRa(dspy.Module):
    """Main IReRa program for vacuum review classification"""
    
    def __init__(self, config: VacuumIReRaConfig):
        super().__init__()
        self.config = config
        
        # Initialize modules
        self.infer_retrieve = InferRetrieve(config)
        self.rank = Rank(config)
        
        # Configuration
        self.rank_skip = config.rank_skip
        self.rank_topk = config.rank_topk
    
    def forward(self, text: str) -> dspy.Prediction:
        # Get initial ranking from InferRetrieve
        prediction = self.infer_retrieve(text)
        labels = prediction.predictions
        
        # Get top-k candidates for ranking
        options = labels[:self.rank_topk]
        
        # Optionally rerank with LM
        if not self.rank_skip:
            reranked = self.rank(text, options).predictions
            
            # Ensure valid predictions and supplement with original ranking
            selected_options = [o for o in reranked if o in options]
            selected_options += [o for o in options if o not in selected_options]
        else:
            selected_options = options
        
        return dspy.Prediction(predictions=selected_options)

# Evaluation metrics
def precision_at_k(gold: List[str], predicted: List[str], k: int) -> float:
    """Calculate Precision at K"""
    if k == 0:
        return 0.0
    top_k_predicted = predicted[:k]
    true_positives = len([item for item in top_k_predicted if item in gold])
    return true_positives / k

def recall_at_k(gold: List[str], predicted: List[str], k: int) -> float:
    """Calculate Recall at K"""
    if len(gold) == 0:
        return 0.0
    top_k_predicted = predicted[:k]
    true_positives = len([item for item in top_k_predicted if item in gold])
    return true_positives / len(gold)

def f1_at_k(gold: List[str], predicted: List[str], k: int) -> float:
    """Calculate F1 at K"""
    prec = precision_at_k(gold, predicted, k)
    rec = recall_at_k(gold, predicted, k)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

# Evaluation function
def evaluate_program(program: VacuumIReRa, examples: List[dspy.Example]) -> Dict[str, float]:
    """Evaluate the program on examples"""
    total_precision_3 = 0
    total_recall_3 = 0
    total_f1_3 = 0
    total_precision_5 = 0
    total_recall_5 = 0
    total_f1_5 = 0
    
    for example in examples:
        prediction = program(example.text)
        predicted_tags = prediction.predictions
        gold_tags = example.label
        
        # Calculate metrics at different k values
        prec_3 = precision_at_k(gold_tags, predicted_tags, 3)
        rec_3 = recall_at_k(gold_tags, predicted_tags, 3)
        f1_3 = f1_at_k(gold_tags, predicted_tags, 3)
        
        prec_5 = precision_at_k(gold_tags, predicted_tags, 5)
        rec_5 = recall_at_k(gold_tags, predicted_tags, 5)
        f1_5 = f1_at_k(gold_tags, predicted_tags, 5)
        
        total_precision_3 += prec_3
        total_recall_3 += rec_3
        total_f1_3 += f1_3
        total_precision_5 += prec_5
        total_recall_5 += rec_5
        total_f1_5 += f1_5
    
    n = len(examples)
    return {
        "precision@3": total_precision_3 / n,
        "recall@3": total_recall_3 / n,
        "f1@3": total_f1_3 / n,
        "precision@5": total_precision_5 / n,
        "recall@5": total_recall_5 / n,
        "f1@5": total_f1_5 / n,
    }

# Main execution
def main():
    # Set up DSPy (you'll need to configure with your LM)
    # dspy.configure(lm=your_language_model)
    
    # Configuration
    config = VacuumIReRaConfig()
    
    # Load data
    print("Loading vacuum review data...")
    train_examples = load_vacuum_data(config.dataset_dev_path)
    test_examples = load_vacuum_data(config.dataset_test_path)
    
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(test_examples)} test examples")
    
    # Create program
    program = VacuumIReRa(config)
    
    # Example prediction
    sample_review = train_examples[0].text
    print(f"\nSample review: {sample_review[:200]}...")
    prediction = program(sample_review)
    print(f"Predicted tags: {prediction.predictions[:3]}")
    print(f"Actual tags: {train_examples[0].label}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_program(program, test_examples[:10])  # Use subset for quick test
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()
