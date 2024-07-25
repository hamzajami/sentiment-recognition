import gradio as gr
import pickle

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(text):
    result = model(text)
    return result[0]['label']

interface = gr.Interface(fn=predict, inputs='text', outputs='label')

if __name__ == "__main__":
    interface.launch()
