
import cv2
import torch
import numpy as np
import pytesseract
import os
import imagehash
from PIL import Image


from transformers import YolosForObjectDetection, YolosFeatureExtractor

# Initialize the YOLOS model
model = YolosForObjectDetection.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize the YOLOS feature extractor
feature_extractor = YolosFeatureExtractor.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")

def detect_license_plate(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.shape[:2]]), threshold=0.9)[0]
    
    print(results)  # Debug print
    
    if 'labels' in results and 'scores' in results and 'boxes' in results:
        license_plates = [(label, score, box) for label, score, box in zip(results['labels'], results['scores'], results['boxes'])]
        if license_plates:
            for plate, score, box in license_plates:
                print(f"License Plate Detected: {plate}, Confidence Score: {score.item()}")
        else:
            print("No license plates detected.")
    else:
        print("Unexpected result structure:", results)

    return license_plates

def detect_license_plate_text(plate_image):
    text = pytesseract.image_to_string(plate_image, config='--psm 8')
    return text.strip()

def deshear(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Detect edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)  # Detect lines

    if lines is not None:
        angles = [np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]]  # Extract angles
        median_angle = np.median(angles)  # Find median angle
        shear_factor = 2 * -np.tan(median_angle)  # Calculate shear factor

        print(median_angle, shear_factor)
        
        (h, w) = image.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])  # Shearing transformation matrix
        nW = w + abs(shear_factor * h)  # New width after shearing
        desheared_image = cv2.warpAffine(image, M, (int(nW), h))  # Apply shearing
        return desheared_image

    return image


def save_plate_image(image, box):
    x_min, y_min, x_max, y_max = map(int, box)
    plate_img = image[y_min:y_max, x_min:x_max]
    img_hash = imagehash.average_hash(Image.fromarray(plate_img))
    save_path = f'plates/{img_hash}.jpg'
    if not os.path.exists(save_path):
        cv2.imwrite(save_path, plate_img)
    return save_path

# Create 'plates' directory if it doesn't exist
if not os.path.exists('plates'):
    os.makedirs('plates')

# Capture video
video = cv2.VideoCapture('example_video.mov')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # deshear the frame
    # desheared_frame = frame
    desheared_frame = deshear(frame)

    # Detect license plates
    plates = detect_license_plate(desheared_frame)
    
    # Draw rectangles around detected license plates on the desheared frame
    for label, score, box in plates:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(desheared_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        save_plate_image(desheared_frame, box)

    
    # Display the desheared frame
    cv2.imshow('Video', desheared_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture once everything is done
video.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
