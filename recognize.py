from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import os
from natsort import natsorted
import torch

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('./trocr-base-printed_license_plates_ocr')

# Set the device to the GPU if available, otherwise stick with CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Loop through all the images in the 'plates' directory
for filename in natsorted(os.listdir('plates')):
    image_path = os.path.join('plates', filename)
    image = Image.open(image_path)

    # Perform OCR on the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Print the filename and the text recognized from the image
    print(filename, generated_text)