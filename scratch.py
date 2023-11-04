import cv2
import numpy as np

def shear_image(img, shear_factor_x=0.0, shear_factor_y=0.0):
    h, w = img.shape[:2]

    # Transformation matrix for shearing
    M = np.array([[1, shear_factor_x, -shear_factor_x * h / 2],
                  [shear_factor_y, 1, -shear_factor_y * w / 2]], dtype=np.float32)

    # Corners of the original image
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])

    # New image dimensions after shearing
    new_corners = np.dot(M[:2, :2], corners.T).T + M[:2, 2]
    x_min, y_min = np.min(new_corners, axis=0)
    x_max, y_max = np.max(new_corners, axis=0)

    # Translation to keep the image centered
    M[0, 2] -= x_min
    M[1, 2] -= y_min

    # Perform the shearing
    sheared_img = cv2.warpAffine(img, M, (int(x_max - x_min), int(y_max - y_min)))
    return sheared_img

def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def transform_image(image):
    cropped = image[3:-3, 3:-3, :]
    scaled = cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    return scaled

    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = gray
    edges = cv2.Canny(blur, 70, 100, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    horizontal_slopes = []
    vertical_slopes = []

    for line in lines:
        for rho, theta in line:
            # Convert the line equation into slope-intercept form
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Find two points on the line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            print(x1, y1, x2, y2)

            # Calculate the slope (if it's not vertical, otherwise skip)
            if (x2 - x1 == 0):
                # vertical_slopes.append(1000)
                continue
                
            slope = (y2 - y1) / (x2 - x1)

            if (slope < 1):
                horizontal_slopes.append(slope) 

            if (slope > 4):
                vertical_slopes.append(slope)
            
    # Average horizontal line slope
    avg_horizontal_slope = np.mean(horizontal_slopes)
    avg_vertical_slope = np.mean(vertical_slopes)

    # Rotate
    rotation_angle_degrees = np.degrees(np.arctan(avg_horizontal_slope))
    rotated  = rotate_image(scaled, rotation_angle_degrees)

    # Shear
    theta = np.arctan((avg_vertical_slope - avg_horizontal_slope) / (1 + avg_horizontal_slope * avg_vertical_slope))
    theta_degrees = 90 - np.degrees(theta)
    theta = np.radians(theta_degrees)
    sheared = shear_image(rotated, -theta, 0)

    # Scale horizontally
    # scaled = cv2.resize(sheared, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC)
    # _, thresh = cv2.threshold(scaled, 144, 255, cv2.THRESH_BINARY_INV)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return sheared
