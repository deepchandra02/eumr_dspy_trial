import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score, f1_score


class ClassificationEvaluator:
    """Handles evaluation of classification performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate_predictions(
        self, true_tags: List[List[str]], pred_tags: List[List[str]], all_tags: Set[str]
    ) -> Dict:
        """Evaluate classification performance"""
        # Convert to binary matrices for sklearn metrics
        tag_list = sorted(list(all_tags))
        
        y_true = []
        y_pred = []
        
        for true_set, pred_set in zip(true_tags, pred_tags):
            true_binary = [1 if tag in true_set else 0 for tag in tag_list]
            pred_binary = [1 if tag in pred_set else 0 for tag in tag_list]
            y_true.append(true_binary)
            y_pred.append(pred_binary)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        
        # Calculate exact match accuracy
        exact_matches = sum(
            1
            for true_set, pred_set in zip(true_tags, pred_tags)
            if set(true_set) == set(pred_set)
        )
        exact_match_accuracy = exact_matches / len(true_tags)
        
        # Calculate partial match metrics
        partial_matches = []
        for true_set, pred_set in zip(true_tags, pred_tags):
            true_set, pred_set = set(true_set), set(pred_set)
            if true_set and pred_set:
                overlap = len(true_set.intersection(pred_set))
                union = len(true_set.union(pred_set))
                partial_matches.append(overlap / union if union > 0 else 0)
            else:
                partial_matches.append(0)
        
        avg_partial_match = np.mean(partial_matches)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "exact_match_accuracy": exact_match_accuracy,
            "avg_partial_match": avg_partial_match,
            "total_samples": len(true_tags),
        }
    
    def print_detailed_results(self, test_results: Dict):
        """Print detailed test results"""
        print("\n" + "=" * 60)
        print("🎯 CLASSIFICATION TEST RESULTS")
        print("=" * 60)
        
        metrics = test_results["metrics"]
        print(f"📈 Performance Metrics:")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"   Average Partial Match: {metrics['avg_partial_match']:.3f}")
        
        print(f"\n📊 Dataset Info:")
        print(f"   Dev samples used for training: {test_results['dev_data_size']}")
        print(f"   Test samples evaluated: {test_results['test_samples']}")
        print(f"   Total unique tags: {test_results['total_tags']}")
        
        print(f"\n🔍 Individual Results:")
        for result in test_results["results"]:
            print(f"\nSample {result['sample_id'] + 1}:")
            print(f"   Text: {result['text'][:80]}...")
            print(f"   True: {result['true_tags']}")
            print(f"   Pred: {result['predicted_tags']}")
            print(
                f"   Match: {'✅' if set(result['true_tags']) == set(result['predicted_tags']) else '❌'}"
            )
            print(f"   Confidence: {result['confidence']:.3f}")