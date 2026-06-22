# sps_genai - GenAI API

A FastAPI project built for the Applied GenAI course, providing NLP and computer vision endpoints.

## Endpoints

- `GET /` - Health check
- `GET /embedding?word=cat` - Returns a word embedding vector using spaCy
- `POST /generate` - Generates text using a Bigram language model
- `POST /predict-image` - Classifies an image using a CNN trained on CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Setup

```bash
conda activate genai
uvicorn main:app --reload
```

## CNN Model

- Architecture: EnhancedCNN (4 conv layers + BatchNorm + Dropout)
- Dataset: CIFAR-10
- Test Accuracy: ~81%
- Train: `python -m app.train`