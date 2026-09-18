import cv2
import numpy as np
import time

# -----------------------------
# 1. Simple Proportional Controller for Stabilization
# -----------------------------
class ProportionalController:
    def __init__(self, kp=0.005):  # ✅ Fixed constructor
        self.kp = kp

    def compute_control(self, error_x, error_y):
        control_x = self.kp * error_x
        control_y = self.kp * error_y
        return control_x, control_y


controller = ProportionalController(kp=0.005)

# -----------------------------
# 2. Open the Webcam Stream
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 for default webcam, change to 1 if external cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise Exception("❌ Webcam initialization failed. Try changing index to 1 or check permissions.")

print("✅ Edge Detection & Drone VIO Stabilization (Webcam Mode). Press 'q' to exit.")

# -----------------------------
# 3. Camera & Stabilization Parameters
# -----------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2
center_y = frame_height // 2

# -----------------------------
# 4. Main Loop with Terminal Output of VIO Data
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Skipping frame.")
        continue

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_frame = frame.copy()

    # If contours are found, calculate the centroid of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
       
        if M["m00"] != 0:
            cx_obj = int(M["m10"] / M["m00"])
            cy_obj = int(M["m01"] / M["m00"])
           
            error_x = center_x - cx_obj
            error_y = center_y - cy_obj
            control_x, control_y = controller.compute_control(error_x, error_y)
           
            # Draw centroid and overlay error/control info
            cv2.circle(output_frame, (cx_obj, cy_obj), 6, (0, 0, 255), -1)
            cv2.putText(output_frame, f"Error X: {error_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(output_frame, f"Error Y: {error_y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(output_frame, f"Ctrl X: {control_x:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output_frame, f"Ctrl Y: {control_y:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           
            # Print to terminal
            print(f"[{time.strftime('%H:%M:%S')}] VIO -> Centroid: ({cx_obj}, {cy_obj}), "
                  f"Error: (X: {error_x}, Y: {error_y}), Control: (X: {control_x:.3f}, Y: {control_y:.3f})")

    # Optionally, draw all contours for debugging
    for idx, contour in enumerate(contours):
        color = (0, 255, 0) if idx % 2 == 0 else (255, 0, 0)
        cv2.drawContours(output_frame, [contour], -1, color, 2)
        for point in contour:
            cv2.circle(output_frame, tuple(point[0]), 1, (0, 0, 255), -1)

    # Draw the frame center crosshair
    cv2.circle(output_frame, (center_x, center_y), 6, (255, 0, 255), -1)
    cv2.line(output_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 255), 1)
    cv2.line(output_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 255), 1)

    # Combine edges with frame
    output = cv2.addWeighted(output_frame, 0.8, edges_colored, 0.4, 0)
    cv2.imshow("Edge Detection & Drone VIO Stabilization", output)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
