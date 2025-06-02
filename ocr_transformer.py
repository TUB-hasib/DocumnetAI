from PIL import Image
import fitz
import sys
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

dir = "./ml_models/donut-docvqa"

# Load processor and model
# processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
# model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
# model.eval()

processor = DonutProcessor.from_pretrained(dir)
model = VisionEncoderDecoderModel.from_pretrained(dir)
model.eval()

def ocr_donut(images):
    print("Performing OCR using Donut...")

    all_pages_data = []

    for page_num, image in enumerate(images):
        # Preprocess image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Prompt: tells model what to do
        task_prompt = "<s_dococr>"
        decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Decode output
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.replace(processor.tokenizer.eos_token, "").strip()
        print(f"Page {page_num + 1} OCR result: {decoded}")

        # Add to results
        all_pages_data.append({
            "page": page_num + 1,
            "text": decoded
        })

    # Save results to JSON
    with open("./data/ocr_output_donut.json", "w", encoding="utf-8") as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)

    print("OCR completed. Output saved to './data/ocr_output_donut.json'.")


def OCR_LayoutLMv2(images):
    print("Performing OCR using Tesseract...")

    all_pages_data = []

    for page_num, image in enumerate(images):
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        page_data = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i]
            if text.strip() == '' or int(ocr_data['conf'][i]) == -1:
                continue  # Skip empty or low-confidence results
            word_info = {
                'text': text,
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': int(ocr_data['conf'][i])
            }
            page_data.append(word_info)

        all_pages_data.append({
            'page': page_num + 1,
            'words': page_data
        })

    # Output the OCR result to JSON file
    with open('./data/ocr_output.json', 'w', encoding='utf-8') as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)

    print("OCR completed. Output saved to './data/ocr_output.json'.")


def convert_pdf_to_images(pdf_path):
    print(f"Converting PDF {pdf_path} to images...")
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"Converted page {page_num + 1} to image.")
    return images





if __name__ == "__main__":

    
    url = './data/cert.pdf'
    if not url:
        print("No PDF file provided.")
        sys.exit(1)
    
    images = convert_pdf_to_images(url)

    for i, img in enumerate(images):
        img.save(f'./data/page_{i + 1}.png')
        print(f"Saved page {i + 1} as image.")
    
    ocr_donut(images)
    # ocr_LayoutLMv2(images)

    # print(images)

    # tessaractocr()
    # print("OCR processing completed.")