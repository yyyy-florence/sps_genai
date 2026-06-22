from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from app.bigram_model import BigramModel

app = FastAPI()

# ---------- 加载 spaCy 模型（用于 embedding）----------
nlp = spacy.load("en_core_web_sm")

# ---------- 初始化 bigram 模型 ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# ---------- 请求体定义 ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 10

# ---------- 端点 ----------
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV and Docker! (with bigram generation)"}

@app.get("/embedding")
def get_embedding(word: str):
    token = nlp(word)
    if not token.has_vector:
        return {"error": f"Word '{word}' not in vocabulary"}
    return {"word": word, "embedding": token.vector.tolist()}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated = bigram_model.generate_text(request.start_word, request.length)
    return {"start_word": request.start_word, "generated_text": generated}

from fastapi import File, UploadFile
from PIL import Image
import io
from app.inference import predict_image

@app.post("/predict-image")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = predict_image(image)
    return result