import json
import os
from typing import List, Dict, Set
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ClassificationTester:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def load_dev_dataset(self) -> List[Dict]:
        """Load only dev dataset for training embeddings"""
        with open("src/data/dataset-dev.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                item["dataset_type"] = "dev"
        return data

    def load_test_sample(self, n_samples: int = 10) -> List[Dict]:
        """Load first n samples from test dataset"""
        with open("src/data/dataset-test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[:n_samples]

    def get_all_tags_from_dev(self, dev_data: List[Dict]) -> Set[str]:
        """Extract all unique tags from dev dataset"""
        all_tags = set()
        for item in dev_data:
            all_tags.update(item["tags"])
        return all_tags

    def setup_pipeline_with_dev_only(self):
        """Train pipeline using only dev dataset"""
        print("🔄 Loading dev dataset only...")
        dev_data = self.load_dev_dataset()
        print(f"📊 Loaded {len(dev_data)} dev reviews")

        print("🧠 Creating embeddings for dev dataset...")
        texts = [item["text"] for item in dev_data]
        embeddings = self.pipeline.create_embeddings_batch(
            texts, task_type="classification"
        )

        print("💾 Storing dev embeddings in vector database...")
        self.pipeline.store_in_vectordb(dev_data, embeddings)

        # Get all available tags from dev set
        all_tags = self.get_all_tags_from_dev(dev_data)
        print(f"📋 Found {len(all_tags)} unique tags in dev set")

        return dev_data, all_tags

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

    def test_classification(self, n_samples: int = 10) -> Dict:
        """Run complete classification test"""

        # Step 1: Setup pipeline with dev data only
        dev_data, all_tags = self.setup_pipeline_with_dev_only()

        # Step 2: Load test samples
        print(f"\n🧪 Loading {n_samples} test samples...")
        test_samples = self.load_test_sample(n_samples)

        # Step 3: Run classification on test samples
        print("🏷️ Classifying test samples...")
        results = []
        true_tags_list = []
        pred_tags_list = []

        for i, sample in enumerate(test_samples):
            print(f"\nSample {i+1}/{len(test_samples)}:")
            print(f"Text: {sample['text'][:100]}...")

            # Classify using the pipeline
            classification_result = self.pipeline.classify_document(
                document=sample["text"],
                target_tags=list(all_tags),  # Use all dev tags as targets
                n_similar=5,
            )

            true_tags = sample["tags"]
            pred_tags = classification_result["predicted_tags"]

            print(f"True tags: {true_tags}")
            print(f"Predicted tags: {pred_tags}")
            print(f"Confidence: {classification_result['confidence']:.3f}")

            # Store for evaluation
            true_tags_list.append(true_tags)
            pred_tags_list.append(pred_tags)

            results.append(
                {
                    "sample_id": i,
                    "text": sample["text"],
                    "true_tags": true_tags,
                    "predicted_tags": pred_tags,
                    "confidence": classification_result["confidence"],
                    "similar_examples": classification_result["similar_examples"][
                        :2
                    ],  # Top 2 similar
                }
            )

        # Step 4: Evaluate performance
        print("\n📊 Evaluating performance...")
        metrics = self.evaluate_predictions(true_tags_list, pred_tags_list, all_tags)

        return {
            "results": results,
            "metrics": metrics,
            "dev_data_size": len(dev_data),
            "test_samples": len(test_samples),
            "total_tags": len(all_tags),
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


# Usage function
def run_classification_test():
    """Main function to run the classification test"""
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    # Import the pipeline class (assuming it's in the same directory)
    from old_approach.main import GeminiClassificationPipeline

    # Initialize pipeline
    pipeline = GeminiClassificationPipeline(api_key=api_key)

    # Initialize tester
    tester = ClassificationTester(pipeline)

    # Run test
    test_results = tester.test_classification(n_samples=10)

    # Print results
    tester.print_detailed_results(test_results)

    return test_results


if __name__ == "__main__":
    results = run_classification_test()
