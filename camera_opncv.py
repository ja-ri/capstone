import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from screeninfo import get_monitors


class image_thread(QThread):

    image_updated = Signal(np.ndarray)  # Signal to indicate image update
    message_updated = Signal(str)
    x_y_update = Signal(np.ndarray)
    original_image_update = Signal(np.ndarray)


    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.width = 0
        self.height = 0
        self.channel = 3
        self.not_stoped = False
        self.caliberate_flag = False
        self.output_caliberated_image = False
        self.value_tplx = 0
        self.value_tply = 0
        self.value_tprx = 0
        self.value_tpry = 0
        self.value_btlx = 0
        self.value_btly = 0
        self.value_btrx = 0
        self.value_btry = 0
        self.thesh_value = 0


    def stop(self):
        self.not_stoped = True
        self.wait()

    def caliberate_on(self):
        self.caliberate_flag = True
    def caliberate_off(self):
        self.caliberate_flag = False

    def run(self):
        while(not self.not_stoped):
            self.main_camcv()
    def emit_message(self,message):
        self.message_updated.emit(message)  # Emit the output message
    def emit_image(self, image):
        self.image_updated.emit(image)  # Emit the output message
    def emit_original_image(self, image):
        self.original_image_update.emit(image)  # Emit the output message
    def emit_x_y(self,message):
        self.x_y_update.emit(message)

    def output_calib_image_ (self,image):
        total_value = [self.value_tplx ,  self.value_tply , self.value_tprx , self.value_tpry , self.value_btlx , self.value_btly , self.value_btrx ,self.value_btry]
        filtered = total_value.count(0)
        if filtered != 8:
            # print("Calibrated image broadcasting")
            # Ensure that the slicing operation is within the bounds of the image
            height, width = image.shape[:2]

            # Calculate the slice dimensions
            start_x = min(self.value_tplx, width)
            start_y = min(self.value_tply, height)
            end_x = min(self.value_tprx, width)
            end_y = min(self.value_btly, height)

            # Crop the image
            cropped_image = image[start_y:end_y, start_x:end_x]
            cropped_image = cv2.resize(cropped_image,(get_monitors()[0].width ,get_monitors()[0].height))
            cropped_image = np.ascontiguousarray(cropped_image)
            return cropped_image
        else:
            # print("Non -caliberated image broadcasting")
            image = np.ascontiguousarray(image)
            return image


    def main_camcv(self):

        print(self.camera_index)
        cap = cv2.VideoCapture(f"/dev/{self.camera_index}")  # Open the selected camera
        
        if not cap.isOpened():
            self.emit_message("Error: Failed to open camera.")
            return

        while not self.not_stoped:
            # print(f"width {width} height {height} channel {channel} stop is {stop}")
            # print(f"not stop {not_stoped}")
            ret, frame = cap.read()  # Read a frame from the camera'
            # print(frame.shape)
            
            if (not ret):
                self.emit_message("Error: Failed to read frame.")
                break
            elif(self.not_stoped):
                self.emit_message("Broadcasting ended")
                break
            # Lets resize to fit in the image

            # print("opencv updated")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Thresholding to isolate bright areas (IR light)
            _, thresh = cv2.threshold(gray, self.thesh_value, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            original_image = thresh
            
            # Iterate through contours
            for contour in contours:
                # Compute centroid of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # print(cX,cY)
                    coordinates = [cX,cY]

                    if (self.caliberate_flag):
                        self.emit_x_y(coordinates)
            
            self.emit_image(self.output_calib_image_(thresh))
            self.emit_original_image(self.output_calib_image_(original_image))
            
        cap.release()

if __name__ == "__main__":
    initialize = image_thread()