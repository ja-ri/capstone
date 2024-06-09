import sys
import threading
from tracemalloc import start
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QMainWindow, QGraphicsScene
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal, QObject, Qt, QTimer
import cv2
from cameras import get_available_cameras
import numpy as np
from camera_opncv import image_thread

class capstone():
    # Initialize the application
    def __init__(self):
        self.loader = QUiLoader()
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = self.loader.load("main.ui")

        # Initialize UI elements
        self.radiobutton = self.window.radiobutton
        self.comboBox = self.window.comboBox
        self.stop_button = self.window.end_button
        self.load_button = self.window.load_button
        self.startbutton = self.window.start_button
        self.save_button = self.window.save_button
        self.textbox = self.window.textbox
        self.slider_box = self.window.slider_box

        # Calibration buttons
        self.calibrate_mode = self.window.caliberate_checkBox
        self.calibrate_TPL = self.window.checkBox_TPL
        self.calibrate_TPR = self.window.checkBox_TPR
        self.calibrate_BTL = self.window.checkBox_BTL
        self.calibrate_BTR = self.window.checkBox_BTR

        # Calibration value boxes
        self.value_tplx = self.window.tplx
        self.value_tplx.setRange(0, 1000)
        self.value_tply = self.window.tply
        self.value_tply.setRange(0, 1000)
        self.value_tprx = self.window.tprx
        self.value_tprx.setRange(0, 1000)
        self.value_tpry = self.window.tpry
        self.value_tpry.setRange(0, 1000)
        self.value_btlx = self.window.btlx
        self.value_btlx.setRange(0, 1000)
        self.value_btly = self.window.btly
        self.value_btly.setRange(0, 1000)
        self.value_btrx = self.window.btrx
        self.value_btrx.setRange(0, 1000)
        self.value_btry = self.window.btry
        self.value_btry.setRange(0, 1000)

        # Slider
        self.slider = self.window.horizontalSlider
        self.slider_box.setValue(self.slider.value())

        # Graphic view
        self.graphicview = self.window.graphicview
        self.graphic_width = self.graphicview.size().width()
        self.graphic_height = self.graphicview.size().height()
        self.graphic_color_type = 3
        self.stop_signal = False

        # Connect UI elements to their respective functions
        self.radiobutton.setChecked(False)
        self.radiobutton.toggled.connect(self.on_radio_button_toggled)
        self.startbutton.clicked.connect(self.run_start)
        self.stop_button.clicked.connect(self.stop_program)
        self.save_button.clicked.connect(self.save_calib_values)
        self.calibrate_mode.stateChanged.connect(self.calibrate_mode_)
        self.calibrate_TPL.stateChanged.connect(self.update_xy_)
        self.calibrate_TPR.stateChanged.connect(self.update_xy_)
        self.slider.valueChanged.connect(self.slider_chaged_)

        # Initialize the OpenCV thread
        self.opencv_thread = image_thread()
        self.opencv_thread.thesh_value = self.slider.value()

    # Function to update slider value
    def slider_chaged_(self):
        self.slider_box.setValue(self.slider.value())
        self.opencv_thread.thesh_value = self.slider.value()

    # Load calibration values from a file
    def load_calib_values(self):
        print("load")
        with open('calibration_data.txt', 'r') as file:
            for i, line in enumerate(file):
                items = [self.value_tplx, self.value_tply, self.value_tprx, self.value_tpry, self.value_btlx, self.value_btly, self.value_btrx, self.value_btry]
                items[i].setValue(int(line.strip()))

    # Save calibration values to a file
    def save_calib_values(self):
        print("save")
        with open('calibration_data.txt', 'w') as file:
            content = f"{self.comboBox.currentText().strip('video')}\n{self.slider_box.value()}\n{self.value_tplx.value()}\n{self.value_tply.value()}\n{self.value_tprx.value()}\n{self.value_tpry.value()}\n{self.value_btlx.value()}\n{self.value_btly.value()}\n{self.value_btrx.value()}\n{self.value_btry.value()}"
            file.write(content)

    # Close event handling
    def closeEvent(self, event):
        self.stop_program()
        event.accept()

    # Stop the program and save the calibration data
    def stop_program(self):
        with open('calibration_data.txt', 'w') as file:
            content = f"{self.comboBox.currentText().strip('video')}\n{self.slider_box.value()}\n{self.value_tplx.value()}\n{self.value_tply.value()}\n{self.value_tprx.value()}\n{self.value_tpry.value()}\n{self.value_btlx.value()}\n{self.value_btly.value()}\n{self.value_btrx.value()}\n{self.value_btry.value()} "
            file.write(content)
        self.update_output_terminal("Exiting application...")
        if self.opencv_thread.isRunning():
            self.opencv_thread.stop()
            self.opencv_thread.wait()
        self.app.quit()

    # Update calibration values based on the selected checkbox
    def value_add_xy_(self, mssg):
        if self.calibrate_TPL.isChecked():
            self.value_tplx.setValue(mssg[0])
            self.value_tply.setValue(mssg[1])
        elif self.calibrate_TPR.isChecked():
            self.value_tprx.setValue(mssg[0])
            self.value_tpry.setValue(mssg[1])
        elif self.calibrate_BTL.isChecked():
            self.value_btlx.setValue(mssg[0])
            self.value_btly.setValue(mssg[1])
        elif self.calibrate_BTR.isChecked():
            self.value_btrx.setValue(mssg[0])
            self.value_btry.setValue(mssg[1])

    # Connect calibration update signal
    def update_xy_(self):
        if self.calibrate_mode.isChecked():
            self.opencv_thread.x_y_update.connect(self.value_add_xy_)

    # Load calibration mode
    def calibrate_load_(self):
        if self.calibrate_load.isChecked():
            self.update_output_terminal("Loading calibration file")
        else:
            self.update_output_terminal("Disable calibration loading")

    # Enable or disable calibration mode
    def calibrate_mode_(self):
        if self.calibrate_mode.isChecked():
            self.opencv_thread.calibrate_on()
            self.update_output_terminal("Calibration mode activated")
        else:
            self.opencv_thread.calibrate_off()
            self.update_output_terminal("Calibration mode deactivated")

    # Stop the camera feed
    def stop_camera_opencv(self):
        if self.comboBox.currentText() != "No camera selected":
            self.opencv_thread.stop()
            self.pygame_thread.stop_pygame_functions()
            self.graphicview.scene().clear()

    # Toggle radio button
    def on_radio_button_toggled(self):
        if self.radiobutton.isChecked():
            self.update_output_terminal("Searching for cameras")
            cameras = get_available_cameras()
            self.comboBox.addItems(cameras)
        else:
            self.comboBox.clear()
            self.comboBox.addItem("No camera selected")
            self.update_output_terminal("stop searching cameras")

    # Run the OpenCV thread
    def run_thread(self):
        current_camera_index = self.window.comboBox.currentIndex()
        camera_name = self.window.comboBox.itemText(current_camera_index)
        self.opencv_thread.camera_index = current_camera_index-1
        self.opencv_thread.height = self.graphic_height
        self.opencv_thread.width = self.graphic_width
        self.update_output_terminal("Video Broadcasting")
        self.opencv_thread.message_updated.connect(self.update_output_terminal)
        self.opencv_thread.image_updated.connect(self.render_graphics)
        self.opencv_thread.start()

    # Update the output terminal with a message
    def update_output_terminal(self, mssg):
        self.textbox.setText(mssg)

    # Start the camera feed
    def run_start(self):
        if self.comboBox.currentText() == "No camera selected":
            self.update_output_terminal("Please select the camera")
        else:
            self.opencv_thread.not_stoped = False
            self.run_thread()

    # Render the camera feed to the graphics view
    def render_graphics(self, image):
        self.image = image
        q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_Grayscale8)
        scene = self.graphicview.scene()
        if scene is None:
            scene = QGraphicsScene()
            self.graphicview.setScene(scene)
        else:
            scene.clear()
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        pixmap_item.setScale(self.graphicview.width() / pixmap_item.boundingRect().width())
        scene.addItem(pixmap_item)

    # Update calibration coordinates
    def update_calibration_coordinates(self):
        self.opencv_thread.value_tplx = int(self.value_tplx.value())
        self.opencv_thread.value_tply = int(self.value_tply.value())
        self.opencv_thread.value_tprx = int(self.value_tprx.value())
        self.opencv_thread.value_tpry = int(self.value_tpry.value())
        self.opencv_thread.value_btlx = int(self.value_btlx.value())
        self.opencv_thread.value_btly = int(self.value_btly.value())
        self.opencv_thread.value_btrx = int(self.value_btrx.value())
        self.opencv_thread.value_btry = int(self.value_btry.value())

# Main thread to start the application
def main_thread():
    start_capston = capstone()
    start_capston.window.show()
    timer = QTimer()
    timer.timeout.connect(start_capston.app.processEvents)
    timer.start(100)
    start_capston.app.exec()
    print("Exiting gracefully!")

if __name__ == "__main__":
    main_thread()
