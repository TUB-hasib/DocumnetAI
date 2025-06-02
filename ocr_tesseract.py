# import pytesseract
from PIL import Image
import fitz
import sys
import json

def OCR_tesseract(images):
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

    
    url = './data/transcript.pdf'
    if not url:
        print("No PDF file provided.")
        sys.exit(1)
    
    images = convert_pdf_to_images(url)

    for i, img in enumerate(images):
        img.save(f'./data/page_{i + 1}.jpg')
        print(f"Saved page {i + 1} as image.")
    
    # OCR_tesseract(images)

    # print(images)

    # tessaractocr()
    # print("OCR processing completed.")