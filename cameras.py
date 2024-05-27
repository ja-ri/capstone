import os
import cv2

def get_available_cameras(max_cameras=2):
    camera_devices = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            print(f"Camera at index {index} is not available.")
        else:
            ret, frame = cap.read()
            if ret:
                camera_devices.append(index)
                print(f"Camera at index {index} is available.")
            cap.release()
    cameras = []
    for i in range(len(camera_devices)):
        
        camera = "camera"+ str(i)
        cameras.append(camera)
    return cameras

if __name__ == "__main__":
    available_cameras = get_available_cameras()
    if available_cameras:
        print("Available cameras:")
        for device_index in available_cameras:
            print(f"Camera {device_index}")
    else:
        print("No cameras detected.")