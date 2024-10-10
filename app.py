import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the fine-tuned model and tokenizer from saved directory
model = BertForSequenceClassification.from_pretrained('./model/fine_tuned_bert_reviews')
tokenizer = BertTokenizer.from_pretrained('./model/fine_tuned_bert_reviews')

# Set device to MPS (if available) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Set model to evaluation mode
model.eval()

# Streamlit App
st.title("Steam Review Classifier")
st.write("Enter a review below, and the model will predict whether the review is Recommended or Not Recommended.")

# Text input from user
review = st.text_area("Enter a review:")

# Prediction button
if st.button("Analyze"):
    if review.strip():  # Ensure the review is not empty
        # Tokenize the review text
        inputs = tokenizer(review, truncation=True, padding=True, max_length=128, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted sentiment (0 = Negative, 1 = Positive)
        prediction = torch.argmax(logits, dim=-1).item()
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display the prediction
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a valid review.")
