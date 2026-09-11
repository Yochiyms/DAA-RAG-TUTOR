from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

reader = PdfReader("daa_book.pdf")
print(f"Number of pages found: {len(reader.pages)}")

text = ""
pages_with_no_text = 0
for i, page in enumerate(reader.pages):
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"
    else:
        pages_with_no_text += 1

print(f"Pages with no text: {pages_with_no_text}")

with open("book.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(text)} characters")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="Prakhar_DAA")

embeddings = model.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("Vectors stored successfully in Prakhar_DAA collection")

# Step 4: Test retrieval
query = "What is dynamic programming?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)
print(results["documents"])