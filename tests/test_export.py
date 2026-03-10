import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading model...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "OxideXLA is an incredibly fast machine learning compiler!"
inputs = tokenizer(text, return_tensors="pt")

print("Exporting...")
try:
    torch.onnx.export(model, (inputs['input_ids'], inputs['attention_mask']), "distilbert.onnx", 
                      input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                      opset_version=14)
    print("Export done")
except Exception as e:
    print(e)
