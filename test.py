from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json

# # Load processor and parsing model (e.g., for receipts)
# model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
# processor = DonutProcessor.from_pretrained(model_id)
# model = VisionEncoderDecoderModel.from_pretrained(model_id)
# model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# def parse_document(image_path):
#     image = Image.open(image_path).convert("RGB")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Donut uses special prompt format
#     task_prompt = "<s_docvqa>"  # use <s_sroie> or your own tag if different
#     decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)
#     pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             pixel_values,
#             decoder_input_ids=decoder_input_ids,
#             max_length=1024,
#             pad_token_id=processor.tokenizer.pad_token_id
#         )

#     output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#     output = output.replace(processor.tokenizer.eos_token, "").strip()
#     print(f"Parsed Output: {output}")
#     print("Output Length:", len(output))
#     print("Output Type:", type(output))

#     try:
#         parsed = json.loads(output)
#         print("Parsed Data:", json.dumps(parsed, indent=2))
#     except json.JSONDecodeError:
#         print("Raw Output (not JSON):", output)

# parse_document("./data/page_1.jpg")  # change to your file


# Load pretrained Donut model (fine-tuned on form-like docs)
model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Load image
image = Image.open("./data/page_1.jpg").convert("RGB")

# Prepare image for Donut
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate output
# task_prompt = "<s_docvqa><s_question>find student's degree and registration number in this form?</s_question><s_answer>"
task_prompt = "<s_docvqa><s_question>Extract all key-value pairs from the form.</s_question><s_answer>"

decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)

# Decode the result
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Extracted schema:\n", result)






# Option 2: Extract Specific Fields Individually (Iterative Prompting)
# If you have specific known fields (e.g., "Name", "DOB", "Phone"), you can loop through them:


# fields = ["Name", "Date of Birth", "Phone Number", "Email"]

# for field in fields:
#     question = f"<s_docvqa><s_question>What is the {field}?</s_question><s_answer>"
#     decoder_input_ids = processor.tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids
#     outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
#     result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#     print(f"{field}: {result}")


# Pro Tip: Save and Reuse the Model Locally
# If you're going to run this often, download and cache the model:


# processor.save_pretrained("./donut_processor")
# model.save_pretrained("./donut_model")
# Then load it later without downloading again:


# processor = DonutProcessor.from_pretrained("./donut_processor")
# model = VisionEncoderDecoderModel.from_pretrained("./donut_model")
