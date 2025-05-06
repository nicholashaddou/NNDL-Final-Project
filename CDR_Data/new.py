
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 1. Choose the pretrained biomedical model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# 2. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Define number of unique BIO tags in your dataset
# For example: ['O', 'B-Chemical', 'I-Chemical', 'B-Disease', 'I-Disease'] → 5 tags
num_labels = 5

# 4. Load the model with the correct output head
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)