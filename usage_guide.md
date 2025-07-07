# Infer-Retrieve-Rank Document Classifier - Usage Guide

This implementation follows the methodology from "DSPy: In-Context Learning for Extreme Multi-Label Classification" to classify documents using OpenAI's text-embeddings-003 and GPT-4.1 models.

## 🏗️ Architecture Overview

The system implements a three-stage classification process:

1. **Infer**: GPT model analyzes document text and generates initial label predictions
2. **Retrieve**: Text embeddings find similar labels in your label space using cosine similarity
3. **Rank**: GPT model re-ranks retrieved labels based on document context

## 📁 Required File Structure

```
eumr_dspy_trial/
├── labels.csv              # Your label data (required)
├── document1.txt          # Text files to classify
├── document2.txt
├── ...
├── embeddings_cache.pkl   # Generated automatically
└── classification_results.json  # Output file
```

## 📊 CSV Format Requirements

Your `labels.csv` must contain exactly these columns:

```csv
label,category,description
"Software Engineering","Technology","Development of software applications and systems"
"Data Analysis","Analytics","Statistical analysis and data interpretation"
"Project Management","Business","Planning and execution of projects"
```

**Important**:

- `label`: The actual label/tag name
- `category`: Broader category this label belongs to
- `description`: Detailed description of what this label represents

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. Prepare Your Data

Place your:

- Text files in the `eumr_dspy_trial/` folder
- `labels.csv` file in the same folder

## 💻 Basic Usage

### Simple Classification

```python
from document_classifier import DocumentClassifier, DocumentClassifierConfig

# Initialize with default settings
classifier = DocumentClassifier()

# Setup and load data
classifier.setup()  # Reads OPENAI_API_KEY from environment
classifier.load_label_data()  # Loads CSV and generates embeddings
classifier.initialize_classifier()

# Classify a single text
result = classifier.classify_text("Your document text here...")
print(f"Predicted labels: {result['predicted_labels']}")

# Classify a file
result = classifier.classify_file("path/to/document.txt")

# Classify all files in folder
results = classifier.classify_folder()
classifier.save_results(results)
```

### Advanced Configuration

```python
# Custom configuration
config = DocumentClassifierConfig()
config.max_final_labels = 3  # Return top 3 labels
config.similarity_threshold = 0.4  # Higher threshold for stricter matching
config.inference_model = "gpt-4o-mini"  # Faster, cheaper model for inference
config.ranking_model = "gpt-4o"  # Best model for final ranking

classifier = DocumentClassifier(config)
```

## ⚙️ Configuration Options

### Model Settings

- `inference_model`: Model for initial label prediction (default: "gpt-4o-mini")
- `ranking_model`: Model for final ranking (default: "gpt-4o")
- `embedding_model`: Embedding model (default: "text-embedding-3-small")

### Classification Settings

- `max_initial_predictions`: Max labels from inference step (default: 10)
- `max_retrieved_labels`: Max labels from retrieval step (default: 20)
- `max_final_labels`: Max final labels returned (default: 5)
- `similarity_threshold`: Minimum cosine similarity for retrieval (default: 0.3)

### File Paths

- `data_folder`: Folder containing your files (default: "eumr_dspy_trial")
- `csv_file`: Name of your labels CSV (default: "labels.csv")

## 📈 Understanding the Output

### Single Document Result

```json
{
  "text": "Document preview (first 200 chars)...",
  "predicted_labels": ["Software Engineering", "Data Analysis"],
  "num_labels": 2,
  "file_path": "eumr_dspy_trial/document1.txt",
  "file_size": 1500
}
```

### Batch Results

The `classification_results.json` contains all classification results with metadata for analysis.

## 🔧 Troubleshooting

### Common Issues

1. **"Missing required columns" error**

   - Ensure your CSV has exactly: `label`, `category`, `description` columns

2. **"No labels retrieved" warning**

   - Lower the `similarity_threshold` in config
   - Check if your label descriptions are detailed enough

3. **Empty predictions**

   - Your text might be too short or unclear
   - Try preprocessing text (remove special characters, ensure proper encoding)

4. **API errors**
   - Verify your OpenAI API key is valid and has sufficient credits
   - Check rate limits if processing many files

### Performance Tips

1. **Embedding Caching**: Embeddings are automatically cached. Delete `embeddings_cache.pkl` to regenerate.

2. **Batch Processing**: For large datasets, process files in smaller batches to avoid API rate limits.

3. **Model Selection**:
   - Use `gpt-4o-mini` for inference (faster, cheaper)
   - Use `gpt-4o` for ranking (better quality)

## 🎯 Example Workflow

```python
# Complete workflow example
def classify_documents():
    # 1. Setup
    config = DocumentClassifierConfig()
    config.max_final_labels = 3  # Want top 3 labels per document

    classifier = DocumentClassifier(config)
    classifier.setup()

    # 2. Load data and initialize
    classifier.load_label_data("eumr_dspy_trial/labels.csv")
    classifier.initialize_classifier()

    # 3. Process all documents
    results = classifier.classify_folder()

    # 4. Save and analyze results
    classifier.save_results(results, "my_results.json")

    # 5. Print summary
    print(f"Classified {len(results)} documents")
    for result in results:
        if "error" not in result:
            print(f"{result['file_path']}: {result['predicted_labels']}")

if __name__ == "__main__":
    classify_documents()
```

## 📝 Notes

- **First run**: Will take longer due to embedding generation
- **Subsequent runs**: Much faster due to embedding caching
- **Cost**: Mainly from GPT API calls (embeddings are cached)
- **Accuracy**: Improves with better label descriptions and more diverse training examples

## 🔄 Next Steps

After getting initial results:

1. **Evaluate accuracy** on a sample of documents
2. **Adjust configuration** based on performance
3. **Refine label descriptions** for better retrieval
4. **Experiment with different models** for cost/quality trade-offs

The system is designed to be modular and easily configurable for different use cases and datasets.
