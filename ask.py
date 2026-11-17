import os
import pickle
import time

from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

INDEX_FILE = "index.pkl"
TOP_K = 4
LLM_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-3.8-flash"]


def load_index():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]


def retrieve(question, model, chunks, embeddings, k=TOP_K):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_vec
    best = scores.argsort()[::-1][:k]
    return [chunks[i] for i in best]


def build_prompt(question, context_chunks):
    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in context_chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Keep the answer short and simple.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate_with_retry(client, prompt, tries=3):
    for model in LLM_MODELS:
        for attempt in range(tries):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except errors.ServerError:
                wait = 2 ** (attempt + 1)
                print(f"[{model}] server busy, retrying in {wait}s...")
                time.sleep(wait)
            except errors.ClientError as e:
                print(f"[{model}] not usable ({e.code}), trying next model...")
                break
    return None


def main():
    chunks, embeddings = load_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print("UIDAI Doc Assistant. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        top_chunks = retrieve(question, embed_model, chunks, embeddings)
        answer = generate_with_retry(client, build_prompt(question, top_chunks))
        if answer is None:
            print("\nLLM is busy right now. Most relevant passages instead:")
            for c in top_chunks:
                print(f"\n--- {c['source']} (page {c['page']}) ---\n{c['text'][:500]}")
            continue
        print("\nAnswer:", answer)
        print("\nSources:")
        for c in top_chunks:
            print(f" - {c['source']} (page {c['page']})")


if __name__ == "__main__":
    main()