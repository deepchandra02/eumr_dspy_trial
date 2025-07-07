"""
Infer-Retrieve-Rank (IReRa) Document Classifier
Implementation based on "DSPy: In-Context Learning for Extreme Multi-Label Classification"

This module implements the three-stage classification process:
1. Infer: Generate initial label predictions from input text
2. Retrieve: Map predictions to actual label space using embeddings
3. Rank: Re-rank retrieved labels using LM reasoning
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI
import dspy
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentClassifierConfig:
    """Configuration class for the document classifier"""

    def __init__(self):
        # File paths
        self.data_folder = "eumr_dspy_trial"
        self.csv_file = "labels.csv"  # Should contain: label, category, description
        self.embeddings_cache = "embeddings_cache.pkl"

        # OpenAI settings
        self.embedding_model = "text-embedding-3-small"  # or text-embedding-3-large
        self.inference_model = "gpt-4o-mini"  # for infer step
        self.ranking_model = "gpt-4o"  # for ranking step

        # Classification settings
        self.max_initial_predictions = 10
        self.max_retrieved_labels = 20
        self.max_final_labels = 5

        # Embedding settings
        self.embedding_dimension = 1536  # for text-embedding-3-small
        self.similarity_threshold = 0.3


class EmbeddingRetriever:
    """Handles embedding generation and retrieval of similar labels"""

    def __init__(self, config: DocumentClassifierConfig):
        self.config = config
        self.client = OpenAI()
        self.label_embeddings = {}
        self.labels_df = None

    def load_labels(self, csv_path: str) -> pd.DataFrame:
        """Load labels from CSV file"""
        logger.info(f"Loading labels from {csv_path}")

        self.labels_df = pd.read_csv(csv_path)
        required_columns = ["label", "category", "description"]

        # Check if required columns exist
        missing_cols = [
            col for col in required_columns if col not in self.labels_df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(self.labels_df)} labels")
        return self.labels_df

    def generate_embeddings(
        self, force_regenerate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate or load cached embeddings for all labels"""
        cache_path = os.path.join(self.config.data_folder, self.config.embeddings_cache)

        # Try to load from cache first
        if not force_regenerate and os.path.exists(cache_path):
            logger.info("Loading embeddings from cache")
            with open(cache_path, "rb") as f:
                self.label_embeddings = pickle.load(f)
            return self.label_embeddings

        logger.info("Generating embeddings for labels...")

        # Prepare text for embedding (combine label, category, and description)
        embedding_texts = []
        label_keys = []

        for _, row in self.labels_df.iterrows():
            # Create rich text representation for better embeddings
            text = f"Label: {row['label']}\nCategory: {row['category']}\nDescription: {row['description']}"
            embedding_texts.append(text)
            label_keys.append(row["label"])

        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(embedding_texts), batch_size):
            batch_texts = embedding_texts[i : i + batch_size]
            logger.info(
                f"Processing embedding batch {i//batch_size + 1}/{(len(embedding_texts) + batch_size - 1)//batch_size}"
            )

            response = self.client.embeddings.create(
                model=self.config.embedding_model, input=batch_texts
            )

            batch_embeddings = [np.array(emb.embedding) for emb in response.data]
            all_embeddings.extend(batch_embeddings)

        # Store embeddings with label keys
        self.label_embeddings = {
            label: embedding for label, embedding in zip(label_keys, all_embeddings)
        }

        # Cache the embeddings
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.label_embeddings, f)

        logger.info(f"Generated and cached {len(self.label_embeddings)} embeddings")
        return self.label_embeddings

    def retrieve_similar_labels(
        self, query_predictions: List[str], top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """Retrieve similar labels based on query predictions"""

        # Generate embeddings for query predictions
        query_embeddings = []
        for pred in query_predictions:
            response = self.client.embeddings.create(
                model=self.config.embedding_model, input=[pred]
            )
            query_embeddings.append(np.array(response.data[0].embedding))

        # Calculate similarities
        label_names = list(self.label_embeddings.keys())
        label_vecs = np.array([self.label_embeddings[label] for label in label_names])

        all_similarities = []

        for query_embedding in query_embeddings:
            query_vec = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_vec, label_vecs)[0]

            # Create (label, similarity) pairs
            label_similarities = list(zip(label_names, similarities))
            all_similarities.extend(label_similarities)

        # Remove duplicates and sort by similarity
        unique_similarities = {}
        for label, sim in all_similarities:
            if label not in unique_similarities or sim > unique_similarities[label]:
                unique_similarities[label] = sim

        # Sort by similarity and filter by threshold
        sorted_labels = sorted(
            unique_similarities.items(), key=lambda x: x[1], reverse=True
        )

        # Filter by threshold and return top_k
        filtered_labels = [
            (label, sim)
            for label, sim in sorted_labels
            if sim >= self.config.similarity_threshold
        ]

        return filtered_labels[:top_k]


# DSPy Signatures for structured prompting
class InferSignature(dspy.Signature):
    """Generate initial label predictions from document text"""

    text = dspy.InputField(desc="Document text to classify")
    labels = dspy.OutputField(
        desc="Comma-separated list of predicted labels/keywords that best describe the document content"
    )


class RankSignature(dspy.Signature):
    """Re-rank retrieved labels based on document text"""

    text = dspy.InputField(desc="Original document text")
    candidate_labels = dspy.InputField(desc="List of candidate labels to rank")
    ranked_labels = dspy.OutputField(
        desc="Top relevant labels ranked by relevance, comma-separated"
    )


class InferRetrieveRank(dspy.Module):
    """Main IReRa classification module"""

    def __init__(self, config: DocumentClassifierConfig, retriever: EmbeddingRetriever):
        super().__init__()
        self.config = config
        self.retriever = retriever

        # Initialize DSPy modules
        self.infer = dspy.ChainOfThought(InferSignature)
        self.rank = dspy.ChainOfThought(RankSignature)

    def extract_labels_from_string(self, label_string: str) -> List[str]:
        """Extract and clean labels from LM output"""
        if not label_string:
            return []

        # Split by common delimiters and clean
        labels = re.split(r"[,;|\n]", label_string)

        # Clean and filter labels
        cleaned_labels = []
        for label in labels:
            cleaned = label.strip().strip("\"'").strip()
            if cleaned and len(cleaned) > 2:  # Filter very short labels
                cleaned_labels.append(cleaned)

        return cleaned_labels[: self.config.max_initial_predictions]

    def forward(self, text: str) -> dspy.Prediction:
        """Main classification pipeline"""

        # Step 1: Infer - Generate initial predictions
        logger.info("Step 1: Generating initial label predictions...")
        infer_result = self.infer(text=text)
        initial_predictions = self.extract_labels_from_string(infer_result.labels)

        logger.info(f"Initial predictions: {initial_predictions}")

        if not initial_predictions:
            logger.warning("No initial predictions generated")
            return dspy.Prediction(labels=[])

        # Step 2: Retrieve - Find similar labels using embeddings
        logger.info("Step 2: Retrieving similar labels...")
        retrieved_labels = self.retriever.retrieve_similar_labels(
            initial_predictions, top_k=self.config.max_retrieved_labels
        )

        if not retrieved_labels:
            logger.warning("No labels retrieved")
            return dspy.Prediction(
                labels=initial_predictions[: self.config.max_final_labels]
            )

        # Prepare candidate labels for ranking
        candidate_labels_text = ", ".join([label for label, _ in retrieved_labels])
        logger.info(f"Retrieved {len(retrieved_labels)} candidate labels")

        # Step 3: Rank - Re-rank retrieved labels
        logger.info("Step 3: Re-ranking labels...")
        rank_result = self.rank(text=text, candidate_labels=candidate_labels_text)

        final_labels = self.extract_labels_from_string(rank_result.ranked_labels)
        final_labels = final_labels[: self.config.max_final_labels]

        logger.info(f"Final labels: {final_labels}")

        return dspy.Prediction(labels=final_labels)


class DocumentClassifier:
    """Main interface for document classification"""

    def __init__(self, config: DocumentClassifierConfig = None):
        self.config = config or DocumentClassifierConfig()
        self.retriever = None
        self.classifier = None
        self.is_initialized = False

    def setup(self, api_key: str = None):
        """Initialize the classifier with OpenAI API"""

        if api_key:
            openai.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        # Configure DSPy with OpenAI models
        inference_lm = dspy.OpenAI(model=self.config.inference_model, max_tokens=500)
        ranking_lm = dspy.OpenAI(model=self.config.ranking_model, max_tokens=300)

        # Use inference model as default, ranking model for ranking step
        dspy.settings.configure(lm=inference_lm)

        logger.info("OpenAI models configured successfully")

    def load_label_data(self, csv_path: str = None):
        """Load and prepare label data"""

        if csv_path is None:
            csv_path = os.path.join(self.config.data_folder, self.config.csv_file)

        # Initialize retriever and load labels
        self.retriever = EmbeddingRetriever(self.config)
        self.retriever.load_labels(csv_path)

        # Generate embeddings
        self.retriever.generate_embeddings()

        logger.info("Label data loaded and embeddings generated")

    def initialize_classifier(self):
        """Initialize the main classifier"""

        if self.retriever is None:
            raise ValueError("Must load label data first using load_label_data()")

        self.classifier = InferRetrieveRank(self.config, self.retriever)
        self.is_initialized = True

        logger.info("Classifier initialized successfully")

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify a single text document"""

        if not self.is_initialized:
            raise ValueError(
                "Classifier not initialized. Call setup(), load_label_data(), and initialize_classifier() first"
            )

        # Run classification
        result = self.classifier(text)

        return {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "predicted_labels": result.labels,
            "num_labels": len(result.labels),
        }

    def classify_file(self, file_path: str) -> Dict[str, Any]:
        """Classify a text file"""

        logger.info(f"Classifying file: {file_path}")

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Add file info to result
        result = self.classify_text(text)
        result["file_path"] = file_path
        result["file_size"] = len(text)

        return result

    def classify_folder(
        self, folder_path: str = None, file_extension: str = ".txt"
    ) -> List[Dict[str, Any]]:
        """Classify all files in a folder"""

        if folder_path is None:
            folder_path = self.config.data_folder

        folder_path = Path(folder_path)

        # Find all text files
        text_files = list(folder_path.glob(f"*{file_extension}"))

        if not text_files:
            logger.warning(f"No {file_extension} files found in {folder_path}")
            return []

        logger.info(f"Found {len(text_files)} files to classify")

        results = []
        for file_path in text_files:
            try:
                result = self.classify_file(str(file_path))
                results.append(result)
                logger.info(f"✓ Classified {file_path.name}")
            except Exception as e:
                logger.error(f"✗ Error classifying {file_path.name}: {e}")
                results.append(
                    {
                        "file_path": str(file_path),
                        "error": str(e),
                        "predicted_labels": [],
                        "num_labels": 0,
                    }
                )

        return results

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str = "classification_results.json",
    ):
        """Save classification results to file"""

        output_path = os.path.join(self.config.data_folder, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage of the document classifier"""

    # Configuration
    config = DocumentClassifierConfig()

    # Initialize classifier
    classifier = DocumentClassifier(config)

    try:
        # Setup (make sure OPENAI_API_KEY is set in environment)
        print("Setting up classifier...")
        classifier.setup()

        # Load label data
        print("Loading label data...")
        classifier.load_label_data()

        # Initialize classifier
        print("Initializing classifier...")
        classifier.initialize_classifier()

        # Classify all files in folder
        print("Classifying documents...")
        results = classifier.classify_folder()

        # Save results
        classifier.save_results(results)

        # Print summary
        print(f"\n=== Classification Complete ===")
        print(f"Processed {len(results)} files")

        for result in results[:3]:  # Show first 3 results
            if "error" not in result:
                print(f"\nFile: {Path(result['file_path']).name}")
                print(f"Labels: {', '.join(result['predicted_labels'])}")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Classification failed: {e}")


if __name__ == "__main__":
    main()
