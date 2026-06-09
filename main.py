from fastapi import FastAPI
import spacy

app = FastAPI()

# 加载 spaCy 模型
nlp = spacy.load("en_core_web_sm")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV and Docker!"}

@app.get("/embedding")
def get_embedding(word: str):
    token = nlp(word)
    if not token.has_vector:
        return {"error": f"Word '{word}' not in vocabulary"}
    return {"word": word, "embedding": token.vector.tolist()}