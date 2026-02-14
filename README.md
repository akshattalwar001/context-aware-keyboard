# LSTM based Context Aware Keybaord

A next word prediction model trained on classic books using LSTM. Give it some text and it will suggest what word comes next.

---

## What it does

- predicts top 3 next words for any input text
- can also generate a full sentence from a starting phrase

---

## Model details

The model takes last 5 words as input and predicts the next word from a vocabulary of 5000 words.

```
Input (5 words)
Embedding layer (128 dim)
LSTM (150 units)
LSTM (100 units)
Dense softmax (5000 outputs)
```

trained for 10 epochs on around 564k sequences. final validation accuracy was around 15%.

---

## Files

```
Nextwordpredictor.ipynb   - main notebook
next_word_model.keras     - saved model
tokenizer.pkl             - saved tokenizer
books_clean.txt           - training data
```

---

## How to run

Install dependencies:

```bash
pip install tensorflow numpy
```

Open the notebook in Jupyter or Google Colab and run all cells. GPU is recommended, was trained on T4 in Colab.

To use the saved model directly:

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

SEQUENCE_LENGTH = 5

def predict_next(seed_text, top_n=3):
    seed_text = seed_text.lower()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = token_list[-SEQUENCE_LENGTH:]
    token_list = pad_sequences([token_list], maxlen=SEQUENCE_LENGTH, padding='pre')
    predictions = model.predict(token_list, verbose=0)[0]
    top_indices = predictions.argsort()[::-1]
    suggestions = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, "")
        if word and word != "<OOV>":
            suggestions.append(word)
        if len(suggestions) == top_n:
            break
    return suggestions

print(predict_next("the old man"))
# ['was', 'had', 'who']
```

---

## Some example outputs

next word prediction:

| input | predictions |
|-------|-------------|
| the old man | was, had, who |
| i want to know the only | time, thing, day |
| it was a dark | man, and, thing |

sentence generation:

| seed | output |
|------|--------|
| he was a very | he was a very little man in the room and the whole man was |
| hello there i wish you | hello there i wish you are not a little thing to be a little thing |

---

## Tech used

- Python
- TensorFlow / Keras
- NumPy
- trained on Google Colab

---

## Things that can be improved

- vocabulary is only 5000 words so many words get mapped to OOV token
- context window of 5 words is quite small
- accuracy is around 15% which is low, more training or bigger model may help
- transformer based approach would probably work better
