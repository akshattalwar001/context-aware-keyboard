from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model = load_model("next_word_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

SEQUENCE_LENGTH = 5

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    seed_text = input.text.lower().strip()

    if not seed_text:
        return {"predictions": ["the", "a", "is"]}

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = token_list[-SEQUENCE_LENGTH:]
    token_list = pad_sequences([token_list], maxlen=SEQUENCE_LENGTH, padding="pre")

    predictions = model.predict(token_list, verbose=0)[0]
    top_indices = predictions.argsort()[::-1]

    suggestions = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, "")
        if word and word != "<OOV>":
            suggestions.append(word)
        if len(suggestions) == 3:
            break

    return {"predictions": suggestions}

app.mount("/", StaticFiles(directory=".", html=True), name="static")