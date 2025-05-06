
            
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load your saved model
model = AutoModelForTokenClassification.from_pretrained("./saved_biobert_ner")
tokenizer = AutoTokenizer.from_pretrained("./saved_biobert_ner")

# Label ID mapping
id2label = {
    0: "O",
    1: "B-CHEMICAL",
    2: "I-CHEMICAL",
    3: "B-DISEASE",
    4: "I-DISEASE"
}

# Streamlit UI
st.title("🧠 Medical Named Entity Recognition (NER) App")

# Text input box
sentence = st.text_area("Enter a biomedical sentence:")

# Predict button
if st.button("Analyze Entities"):
    if sentence:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [id2label[p.item()] for p in predictions[0]]

        st.subheader("Detected Entities:")
        for token, label in zip(tokens, labels):
            if label != "O":
                st.write(f"**{token}** → {label}")
    else:
        st.warning("Please enter some text first.")