
from transformers import pipeline
import pickle


model = pipeline('sentiment-analysis')

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")
