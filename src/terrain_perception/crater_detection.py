import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
img = cv2.imread(str(PROJECT_ROOT / "assets" / "sample_maps" / "mar3.jpg"))
if img is None:
    print("Error: Image not loaded. Check the file path.")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
kernel = np.ones((5, 5), np.uint8)
gray_blurred = cv2.morphologyEx(gray_blurred, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(gray_blurred, 50, 150)
dp = 1.1  
minDist = 30
param1 = 50
param2 = 30
minRadius = 20
maxRadius = 80

circles = cv2.HoughCircles(
    gray_blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
    param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
)
output = img.copy()

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    filtered_circles = []

    for (x, y, r) in circles:
        if 20 <= r <= 80:  
            if 100 <= x <= img.shape[1] - 100 and 100 <= y <= img.shape[0] - 100:
                filtered_circles.append((x, y, r))

    # Draw filtered circles
    for (x, y, r) in filtered_circles:
        cv2.circle(output, (x, y), r, (255, 0, 255), 2)  
        cv2.circle(output, (x, y), 2, (0, 0, 255), 2)    
else:
    print("No circles detected with the current parameters.")


cv2.imshow('Detected Circles', output)
cv2.imshow('Blurred Gray Image', gray_blurred)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
