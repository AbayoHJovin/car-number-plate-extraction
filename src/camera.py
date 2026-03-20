# src/camera.py
"""
1.11 Camera Validation
Minimal script to verify webcam access.
"""
import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[camera] Error: Camera not found.")
        return
    
    print("[camera] Camera active. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Validation (1.11)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
