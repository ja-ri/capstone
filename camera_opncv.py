import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from screeninfo import get_monitors

class image_thread(QThread):
    # Define signals to communicate with the main thread
    image_updated = Signal(np.ndarray)  # Signal to indicate image update
    message_updated = Signal(str)  # Signal to send messages
    x_y_update = Signal(np.ndarray)  # Signal to send coordinates
    original_image_update = Signal(np.ndarray)  # Signal to send original image

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index  # Index of the camera to use
        self.width = 0  # Width of the captured image
        self.height = 0  # Height of the captured image
        self.channel = 3  # Number of channels in the image
        self.not_stoped = False  # Flag to control the thread execution
        self.calibrate_flag = False  # Flag to indicate calibration mode
        self.output_calibrated_image = False  # Flag to indicate if calibrated image should be output
        self.value_tplx = 0  # Top-left x-coordinate for calibration
        self.value_tply = 0  # Top-left y-coordinate for calibration
        self.value_tprx = 0  # Top-right x-coordinate for calibration
        self.value_tpry = 0  # Top-right y-coordinate for calibration
        self.value_btlx = 0  # Bottom-left x-coordinate for calibration
        self.value_btly = 0  # Bottom-left y-coordinate for calibration
        self.value_btrx = 0  # Bottom-right x-coordinate for calibration
        self.value_btry = 0  # Bottom-right y-coordinate for calibration
        self.thesh_value = 0  # Threshold value for image processing

    def stop(self):
        # Stop the thread execution
        self.not_stoped = True
        self.wait()

    def calibrate_on(self):
        # Enable calibration mode
        self.calibrate_flag = True

    def calibrate_off(self):
        # Disable calibration mode
        self.calibrate_flag = False

    def run(self):
        # Main loop to process the camera feed
        while not self.not_stoped:
            self.main_camcv()

    def emit_message(self, message):
        # Emit a message to the main thread
        self.message_updated.emit(message)

    def emit_image(self, image):
        # Emit an image to the main thread
        self.image_updated.emit(image)

    def emit_original_image(self, image):
        # Emit the original image to the main thread
        self.original_image_update.emit(image)

    def emit_x_y(self, coordinates):
        # Emit x and y coordinates to the main thread
        self.x_y_update.emit(coordinates)

    def output_calib_image_(self, image):
        # Output a calibrated image based on the specified coordinates
        total_value = [
            self.value_tplx, self.value_tply, self.value_tprx,
            self.value_tpry, self.value_btlx, self.value_btly,
            self.value_btrx, self.value_btry
        ]
        filtered = total_value.count(0)
        if filtered != 8:
            height, width = image.shape[:2]
            start_x = min(self.value_tplx, width)
            start_y = min(self.value_tply, height)
            end_x = min(self.value_tprx, width)
            end_y = min(self.value_btly, height)
            cropped_image = image[start_y:end_y, start_x:end_x]
            cropped_image = cv2.resize(cropped_image, (get_monitors()[0].width, get_monitors()[0].height))
            cropped_image = np.ascontiguousarray(cropped_image)
            return cropped_image
        else:
            image = np.ascontiguousarray(image)
            return image

    def main_camcv(self):
        # Main function to capture and process the camera feed
        cap = cv2.VideoCapture(self.camera_index)  # Open the selected camera
        
        if not cap.isOpened():
            self.emit_message("Error: Failed to open camera.")
            return

        while not self.not_stoped:
            ret, frame = cap.read()  # Read a frame from the camera
            
            if not ret:
                self.emit_message("Error: Failed to read frame.")
                break
            elif self.not_stoped:
                self.emit_message("Broadcasting ended.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            
            # Thresholding to isolate bright areas (IR light)
            _, thresh = cv2.threshold(gray, self.thesh_value, 255, cv2.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            original_image = thresh  # Save the thresholded image
            
            # Iterate through the contours
            for contour in contours:
                # Compute centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    coordinates = [cX, cY]
                    if self.calibrate_flag:
                        self.emit_x_y(coordinates)
            
            # Emit the processed and original images
            self.emit_image(self.output_calib_image_(thresh))
            self.emit_original_image(self.output_calib_image_(original_image))
            
        cap.release()

if __name__ == "__main__":
    initialize = image_thread()
