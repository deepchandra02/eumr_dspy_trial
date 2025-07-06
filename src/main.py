import json
import os
import time
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

# Configure Gemini API (set your API key as environment variable GEMINI_API_KEY)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        print(f"Generating embeddings for {len(input)} documents...")

        for i, doc in enumerate(tqdm(input, desc="Creating embeddings")):
            try:
                result = client.models.embed_content(
                    model="text-embedding-004",
                    contents=doc,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    ),
                )
                embeddings.append(result.embeddings[0].values)

                # Rate limiting to avoid quota exhaustion
                time.sleep(0.5)

                if (i + 1) % 10 == 0:
                    print(f"✓ Processed {i + 1}/{len(input)} embeddings")
            except Exception as e:
                print(f"Error creating embedding for document {i}: {e}")
                raise

        print(f"✓ Successfully created {len(embeddings)} embeddings")
        return embeddings


print("🚀 Starting document processing pipeline...")

# Load your data
data_file = "src/data/raw-data/dataset-dev.json"
print(f"📁 Loading data from {data_file}")
with open(data_file) as f:
    data = json.load(f)

# Use smaller sample for testing to avoid quota limits
data = data[:20]
print(f"✓ Using sample of {len(data)} documents for testing")

# Create data folders if they don't exist
os.makedirs("src/data/chunked-data", exist_ok=True)
os.makedirs("src/data/embeddings_db", exist_ok=True)
print("✓ Created necessary directories")

# Generate chunked filename
base_filename = os.path.splitext(os.path.basename(data_file))[0]
chunked_filename = f"src/data/chunked-data/chunked-{base_filename}.json"

# Check if chunked data already exists
if os.path.exists(chunked_filename):
    print(f"📋 Loading existing chunked data from {chunked_filename}")
    with open(chunked_filename) as f:
        chunked_data = json.load(f)
    # Filter to match our sample size
    chunked_data = chunked_data[: len(data) * 2]  # Approximate chunks per document
    print(f"✓ Loaded {len(chunked_data)} chunks")
else:
    print(f"✂️  Chunking data and saving to {chunked_filename}")

    # Set up the splitter
    chunk_size = 1000
    chunk_overlap = 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(f"⚙️  Configured chunker: size={chunk_size}, overlap={chunk_overlap}")

    # Chunk each document and keep the tags
    chunked_data = []
    print("🔪 Processing documents for chunking...")
    for item in tqdm(data, desc="Chunking documents"):
        chunks = splitter.split_text(item["text"])
        for chunk in chunks:
            chunked_data.append({"text": chunk, "tags": item["tags"]})

    print(f"✓ Created {len(chunked_data)} chunks from {len(data)} documents")

    # Save chunked data
    with open(chunked_filename, "w") as f:
        json.dump(chunked_data, f, indent=2)
    print(f"💾 Saved chunked data to {chunked_filename}")

# Set up embeddings database
persist_directory = "src/data/embeddings_db"
collection_name = f"embeddings_{base_filename}_sample"
print(f"🗄️  Setting up vector database: {persist_directory}/{collection_name}")

# Check if embeddings database already exists
chroma_client = chromadb.PersistentClient(path=persist_directory)
existing_collections = [col.name for col in chroma_client.list_collections()]

if collection_name in existing_collections:
    print(f"📊 Loading existing embeddings from {persist_directory}/{collection_name}")
    vectordb = chroma_client.get_collection(
        name=collection_name, embedding_function=GeminiEmbeddingFunction()
    )
    print(f"✓ Loaded existing collection with {vectordb.count()} documents")
else:
    print(
        f"🧠 Creating new embeddings and saving to {persist_directory}/{collection_name}"
    )
    print("⚠️  This may take several minutes with rate limiting...")

    # Create new collection
    vectordb = chroma_client.create_collection(
        name=collection_name, embedding_function=GeminiEmbeddingFunction()
    )
    print("✓ Created new ChromaDB collection")

    # Prepare data for embedding
    texts = [item["text"] for item in chunked_data]
    metadatas = [{"tags": str(item["tags"])} for item in chunked_data]
    ids = [str(i) for i in range(len(chunked_data))]

    print(f"📝 Preparing {len(texts)} documents for embedding...")

    # Add documents to collection (this triggers embedding generation)
    print("🔄 Adding documents to ChromaDB (generating embeddings)...")
    vectordb.add(documents=texts, metadatas=metadatas, ids=ids)
    print("✓ Successfully added all documents to vector database")

print(f"🎉 Vector database ready with {vectordb.count()} documents")
print("✅ Pipeline completed successfully!")
