from transformers import DonutProcessor, VisionEncoderDecoderModel
import os

def main():
    # Model ID from Hugging Face
    # model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
    model_id = "naver-clova-ix/donut-base"

    # Local folder where you want to save the model and processor
    save_directory = "./ml_models/donut-docvqa"
    

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load from Hugging Face (will be cached if already downloaded)
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Save both locally
    processor.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print(f"Donut model and processor saved to: {save_directory}")


if __name__ == "__main__":
    main()
