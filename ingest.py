import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    print("❌ Error: OPENAI_API_KEY not found in .env file!")
    exit()

# 2. Setup Chroma with OpenAI Embedding Function
# We specify the model here so Chroma knows how to vectorize the text
openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./chroma_db")

# 3. Create (or Re-create) the Collection
# Note: If you already had a collection, you should delete the 'chroma_db' 
# folder first to ensure you don't mix different embedding models.
collection = client.get_or_create_collection(
    name="patient_records",
    embedding_function=openai_ef
)

# 4. Load and Process CSV
CSV_FILE = "patient_phi.csv"

if not os.path.exists(CSV_FILE):
    print(f"❌ Error: {CSV_FILE} not found!")
else:
    df = pd.read_csv(CSV_FILE)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Clean the text and create the searchable content
        content = (f"Patient: {row['name']}. "
                   f"Diagnosis: {row['diagnosis']}. "
                   f"Prescription: {row['prescription']}. "
                   f"Notes: {row['notes']}")
        
        documents.append(content)
        metadatas.append({"patient_name": row['name']})
        ids.append(f"id_{index}")

    # 5. Add to Database
    # Because we gave the collection 'openai_ef' above, 
    # Chroma handles the API calls to OpenAI automatically here.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Success! Ingested {len(documents)} records into chroma_db using OpenAI.")