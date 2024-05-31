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
                camera = "video" + str(index)
                camera_devices.append(camera)
                print(f"Camera at index {index} is available.")
            cap.release()
    

    return camera_devices

if __name__ == "__main__":
    available_cameras = get_available_cameras()
    if available_cameras:
        print("Available cameras:")
        for device_name in available_cameras:
            print(f"/dev/{device_name}")
    else:
        print("No cameras detected.")