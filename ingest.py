import chromadb
import pandas as pd
import os

# 1. Setup Chroma Persistence
# This creates the 'chroma_db' folder your main.py is looking for
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="patient_records")

# 2. Load the CSV file
if not os.path.exists("patient_phi.csv"):
    print("❌ Error: patient_phi.csv not found!")
else:
    df = pd.read_csv("patient_phi.csv")

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Create a rich text string for the AI to "read"
        content = f"Patient: {row['name']}. Diagnosis: {row['diagnosis']}. Prescription: {row['prescription']}. Notes: {row['notes']}"
        documents.append(content)
        metadatas.append({"patient_name": row['name']})
        ids.append(f"id_{index}")

    # 4. Add to the database
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Success! Ingested {len(documents)} records into chroma_db.")