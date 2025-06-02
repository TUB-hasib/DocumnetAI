from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json

# Load processor and parsing model (e.g., for receipts)
model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def parse_document(image_path):
    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Donut uses special prompt format
    task_prompt = "<s_docvqa>"  # use <s_sroie> or your own tag if different
    decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=1024,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    output = output.replace(processor.tokenizer.eos_token, "").strip()
    print(f"Parsed Output: {output}")
    print("Output Length:", len(output))
    print("Output Type:", type(output))

    try:
        parsed = json.loads(output)
        print("Parsed Data:", json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("Raw Output (not JSON):", output)

parse_document("./data/page_1.jpg")  # change to your file
