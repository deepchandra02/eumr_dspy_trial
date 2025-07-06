import os
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

# Import our modular components
from data_loader import DataLoader
from embedding_storage import EmbeddingStorage
from gemini_embeddings import GeminiEmbeddings
from old_approach.vector_database import VectorDatabase
from classifier import DocumentClassifier
from evaluation import ClassificationEvaluator

# Load environment variables
load_dotenv()


class GeminiClassificationPipeline:
    """Main pipeline for Gemini-based document classification"""

    def __init__(self, api_key: str, collection_name: str = "vacuum_reviews"):
        """Initialize the classification pipeline"""
        self.api_key = api_key
        self.collection_name = collection_name

        # Initialize components
        self.data_loader = DataLoader()
        self.embedding_storage = EmbeddingStorage()
        self.gemini_embeddings = GeminiEmbeddings(api_key)
        self.vector_db = VectorDatabase(collection_name, reset_collection=False)
        self.classifier = DocumentClassifier(
            genai.Client(api_key=api_key), self.vector_db, self.gemini_embeddings
        )
        self.evaluator = ClassificationEvaluator()

    def setup_embeddings(self, force_recreate: bool = False):
        """Setup embeddings - load from storage or create new ones"""
        print("🔄 Setting up embeddings...")

        # Load data
        data = self.data_loader.load_datasets()
        print(f"📊 Loaded {len(data)} total documents")

        # Check if we need to create embeddings
        if not force_recreate and self.embedding_storage.embeddings_exist(data):
            print("📂 Loading existing embeddings...")
            embeddings, metadata = self.embedding_storage.load_embeddings()
            print(f"✅ Loaded {len(embeddings)} embeddings from storage")
        else:
            print("🧠 Creating new embeddings...")
            texts = [item["text"] for item in data]
            embeddings = self.gemini_embeddings.create_embeddings_batch(texts)

            # Save embeddings for future use
            self.embedding_storage.save_embeddings(data, embeddings)
            print("💾 Embeddings saved to storage")

        # Store in vector database
        print("💾 Storing embeddings in vector database...")
        self.vector_db.store_embeddings(data, embeddings)

        return data, embeddings

    def run_classification_test(self, n_samples: int = 10) -> Dict:
        """Run classification test using dev data for training and test data for evaluation"""
        print("\n🧪 Running Classification Test")
        print("=" * 50)

        # Setup with dev data only
        print("📚 Setting up pipeline with dev data...")
        dev_data = self.data_loader.load_dev_dataset()

        # Check if embeddings exist for dev data
        if self.embedding_storage.embeddings_exist(dev_data):
            print("📂 Using existing dev embeddings...")
            # Vector database should already have the data if embeddings exist
            collection_info = self.vector_db.get_collection_info()
            if collection_info["count"] == 0:
                print("📂 Loading embeddings from storage...")
                embeddings, _ = self.embedding_storage.load_embeddings()
                self.vector_db.store_embeddings(dev_data, embeddings)
        else:
            print("🧠 Creating embeddings for dev data...")
            texts = [item["text"] for item in dev_data]
            embeddings = self.gemini_embeddings.create_embeddings_batch(texts)
            self.embedding_storage.save_embeddings(dev_data, embeddings)
            self.vector_db.store_embeddings(dev_data, embeddings)

        # Get all tags from dev data
        all_tags = self.data_loader.get_all_unique_tags(dev_data)
        print(f"📋 Found {len(all_tags)} unique tags in dev set")

        # Load test samples
        print(f"\n🎯 Loading {n_samples} test samples...")
        test_samples = self.data_loader.load_test_sample(n_samples)

        # Run classification
        print("🏷️ Running classification...")
        results = []
        true_tags_list = []
        pred_tags_list = []

        for i, sample in enumerate(test_samples):
            print(f"\nSample {i+1}/{len(test_samples)}:")
            print(f"Text: {sample['text'][:100]}...")

            # Classify
            classification_result = self.classifier.classify_document(
                document=sample["text"],
                target_tags=list(all_tags),
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
                    "similar_examples": classification_result["similar_examples"][:2],
                }
            )

        # Evaluate performance
        print("\n📊 Evaluating performance...")
        metrics = self.evaluator.evaluate_predictions(
            true_tags_list, pred_tags_list, all_tags
        )

        test_results = {
            "results": results,
            "metrics": metrics,
            "dev_data_size": len(dev_data),
            "test_samples": len(test_samples),
            "total_tags": len(all_tags),
        }

        # Print detailed results
        self.evaluator.print_detailed_results(test_results)

        return test_results

    def classify_single_document(
        self, document: str, target_tags: List[str] = None
    ) -> Dict:
        """Classify a single document"""
        print(f"\n🏷️ Classifying document: {document[:100]}...")

        result = self.classifier.classify_document(document, target_tags)

        print(f"Predicted tags: {result['predicted_tags']}")
        print(f"Confidence: {result['confidence']:.3f}")

        return result

    def get_pipeline_info(self) -> Dict:
        """Get information about the current pipeline setup"""
        storage_info = self.embedding_storage.get_storage_info()
        vector_info = self.vector_db.get_collection_info()

        return {
            "embedding_storage": storage_info,
            "vector_database": vector_info,
            "components_loaded": {
                "data_loader": True,
                "embedding_storage": True,
                "gemini_embeddings": True,
                "vector_db": True,
                "classifier": True,
                "evaluator": True,
            },
        }


def main():
    """Main function to run the classification pipeline"""
    print("🚀 Starting Gemini Classification Pipeline")
    print("=" * 50)

    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    # Initialize pipeline
    pipeline = GeminiClassificationPipeline(api_key=api_key)

    # Show pipeline info
    print("\n📋 Pipeline Information:")
    info = pipeline.get_pipeline_info()
    print(f"Embedding storage: {info['embedding_storage']['exists']}")
    print(f"Vector database: {info['vector_database']['status']}")

    # Run classification test (will handle embeddings internally)
    print("\n" + "=" * 50)
    test_results = pipeline.run_classification_test(n_samples=10)

    # Example single document classification
    print("\n" + "=" * 50)
    print("🏷️ Single Document Classification Example")
    print("=" * 50)

    test_document = (
        "This vacuum cleaner has terrible suction and the motor is very loud"
    )
    target_tags = [
        "suction_issues",
        "noise_problems",
        "motor_problems",
        "negative_review",
    ]

    result = pipeline.classify_single_document(test_document, target_tags)

    print(f"\nDocument: {test_document}")
    print(f"Predicted tags: {result['predicted_tags']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Tag frequencies from similar docs: {result['tag_frequencies']}")

    print("\nMost similar examples:")
    for i, example in enumerate(result["similar_examples"][:2]):
        print(f"{i+1}. Similarity: {example['similarity']:.3f}")
        print(f"   Text: {example['text']}")
        print(f"   Tags: {example['tags']}")

    print("\n✅ Pipeline execution completed!")
    return pipeline, test_results


if __name__ == "__main__":
    pipeline, results = main()
