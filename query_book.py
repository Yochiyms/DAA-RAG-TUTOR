import chromadb
from sentence_transformers import SentenceTransformer

# Connect to your existing vector database (no need to re-process the PDF)
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="Prakhar_DAA")

def retrieve_relevant_chunks(query, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"][0]

if __name__ == "__main__":
    print("Prakhar_DAA Retrieval Test — type 'exit' to quit\n")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break
        
        chunks = retrieve_relevant_chunks(query)
        
        print(f"\n--- Top {len(chunks)} relevant passages ---")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[Passage {i}]")
            print(chunk[:400])
            print("...")
        print("\n" + "="*50 + "\n")