import cv2
import numpy as np
import os
import shutil
import torch
import yolov5

# load model
model = yolov5.load('keremberke/yolov5m-license-plate')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def detect_license_plate(image):
    results = model(image)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    license_plates = [(label, score, box) for label, score, box in zip(categories, scores, boxes)]
    if license_plates:
        for plate, score, box in license_plates:
            print(f"License Plate Detected: {plate}, Confidence Score: {score.item()}")
    else:
        print("No license plates detected.")

    return license_plates

def save_plate_image(frame_index, plate_index, image, box):
    x_min, y_min, x_max, y_max = map(int, box)
    width, height = x_max - x_min, y_max - y_min
    if 80 < width < 120 and 30 < height < 100:
        plate_img = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(f'plates/{frame_index}_{plate_index}.jpg', plate_img)
    return None

# Create 'plates' directory if it doesn't exist
if not os.path.exists('plates'):
    os.makedirs('plates')
else:
    # Recreate the directory
    shutil.rmtree('plates')
    os.makedirs('plates')

# Capture video
video = cv2.VideoCapture('example_video.mov')

frame_index = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    plates = detect_license_plate(frame)
    for plate_index, (label, score, box) in enumerate(plates):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        save_plate_image(frame_index, plate_index, frame, box)

    frame_index += 1

# Release the capture once everything is done
video.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
