import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

preprocessed_dir = 'plates/preprocessed/'
os.makedirs(preprocessed_dir, exist_ok=True)

def deshear_image(image):
    rows, cols = image.shape[:2]
    
    # Estimate the vertical shearing angle. Adjust experimentally.
    shear_angle = 0.2  # Example value; adjust as necessary
    
    # Transformation matrix for vertical shearing
    M = np.float32([[1, 0, 0],
                    [shear_angle, 1, 0]])
    
    # Apply affine transformation
    desheared = cv2.warpAffine(image, M, (cols, rows + int(shear_angle * cols)))
    
    return desheared

# Iterate through all files in the 'plates' directory
for filename in os.listdir('plates'):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full path to the image
        image_path = os.path.join('plates', filename)
        
        # Open the image
        image = Image.open(image_path)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply the deshearing function to the thresholded image
        image = deshear_image(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )

        # thresh = cv2.bitwise_not(thresh)


        # gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.medianBlur(gray, 3)
        # # perform otsu thresh (using binary inverse since opencv contours work better with white text)
        # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # # cv2.imshow("Otsu", thresh)
        # # cv2.waitKey(0)
        # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        # # apply dilation 
        # dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
        # #cv2.imshow("dilation", dilation)
        # #cv2.waitKey(0)
        # find contours
        try:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # print(sorted_contours)

        cv2.imwrite(os.path.join(preprocessed_dir, filename), thresh)

        # create copy of image
        im2 = thresh.copy()

        plate_num = ""
        # loop through contours and find letters in license plate
        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = im2.shape




            
            # if height of box is not a quarter of total height then skip
            # if height / float(h) > 6: continue
            # ratio = h / float(w)
            # # if height to width ratio is less than 1.5 skip
            # if ratio < 1.5: continue
            area = h * w
            # # if width is not more than 25 pixels skip
            # if width / float(w) > 15: continue
            # # if area is less than 100 pixels skip
            # print(w, h, area)
            if (h > w): continue
            if area < 200: continue


            # draw the rectangle
            rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            if roi is not None and roi.size > 0:
                # roi = cv2.medianBlur(roi, 5)
                cv2.imshow("ROI", roi)
                cv2.waitKey(0)
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11 --oem 3')
                print(text)
                plate_num += text
            else:
                print("ROI is empty.")
           
           

        # Run OCR
        # text = pytesseract.image_to_string(
        #     thresh, 
        #     config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # ).strip()

        # Output the filename and the detected text
        print(f"File: {filename}, Detected Text: {plate_num}")
