from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Load image
image = Image.open("./data/page_1.jpg")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Inference
outputs = model(**inputs)
