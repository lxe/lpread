from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

# Settings for the resulting image
margin = 10
text_height = 50  # Estimate; will be adjusted per image
font = ImageFont.load_default()

# Store image information
image_info = []

for filename in os.listdir('plates'):
    image_path = os.path.join('plates', filename)
    image = Image.open(image_path)

    # OCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Print the text
    print(filename, generated_text)
    
    # Calculate text size
    draw = ImageDraw.Draw(image)
    text_size = draw.textsize(generated_text, font=font)
    text_height = text_size[1] + margin
    
    # Store image size and text
    image_info.append((image, generated_text, text_height))

# Calculate the resulting image size
resulting_image_height = sum(image.height + info[2] + margin for info in image_info) + margin
resulting_image_width = max(image.width for image, _, _ in image_info) + 2 * margin

# Create a blank image
result_image = Image.new('RGB', (resulting_image_width, resulting_image_height), (255, 255, 255))
draw = ImageDraw.Draw(result_image)

current_height = margin

for image, text, text_height in image_info:
    # Place the image on the result image
    result_image.paste(image, (margin, current_height))
    
    # Draw the text below the image
    text_position = (margin, current_height + image.height + 10)
    draw.text(text_position, text, fill=(0, 0, 0), font=font)
    
    # Update the height
    current_height += image.height + text_height + margin

# Save the resulting image
result_image.save('result_image.jpg')
