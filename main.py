import sys
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QMainWindow,QGraphicsScene
from PySide6.QtGui import QPixmap,QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal,Qt
import cv2
from cameras import get_available_cameras
import numpy as np
from camera_opncv import image_thread
from pygame_opencv import PyGameMouse_thread
from pygame_opencv_ir import PyGameIR_thread

class capstone():

    def __init__(self):
        self.loader = QUiLoader()
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = self.loader.load("main.ui")

    # ---------- variables---------------
        # self.image = np.ndarray()
    # ---------- Button Initialzation --------------------------
        self.radiobutton = self.window.radiobutton
        self.comboBox = self.window.comboBox
        self.stop_button = self.window.stop_button
        self.startbutton = self.window.start_button
        self.textbox = self.window.textbox
        self.pygamebutton = self.window.launchpybutton
        self.comboBox_pygame = self.window.comboBox_pygame
        self.slider_box = self.window.slider_box
        # ---- Caliberation buttons
        self.caliberate_mode = self.window.caliberate_checkBox
        self.caliberate_load = self.window.caliberate_load
        self.caliberate_save = self.window.caliberate_save
        self.caliberate_TPL = self.window.checkBox_TPL
        self.caliberate_TPR = self.window.checkBox_TPR
        self.caliberate_BTL = self.window.checkBox_BTL
        self.caliberate_BTR = self.window.checkBox_BTR
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
        self.slider = self.window.horizontalSlider
        self.slider_box.setValue(self.slider.value())
    # ---------- Graphic Box --------------------
        self.graphicview = self.window.graphicview
        self.graphic_width = self.graphicview.size().width()
        self.graphic_height = self.graphicview.size().height()
        self.graphic_color_type = 3
        self.stop_signal = False
    # ---------- connection --------------------------
        self.radiobutton.setChecked(False)
        self.caliberate_load.setChecked(False)
        self.radiobutton.toggled.connect(self.on_radio_button_toggled)
        self.startbutton.clicked.connect(self.run_start)
        self.stop_button.clicked.connect(self.stop_camera_opencv)
        self.pygamebutton.clicked.connect(self.run_pygame)
        self.comboBox_pygame.currentTextChanged.connect(self.select_bcombo_mode)
        self.caliberate_mode.stateChanged.connect(self.caliberate_mode_)
        self.caliberate_load.stateChanged.connect(self.load_calib_values)
        self.caliberate_TPL.stateChanged.connect(self.update_xy_)
        self.caliberate_TPR.stateChanged.connect(self.update_xy_)
        self.caliberate_save.stateChanged.connect(self.save_calib_values)
        self.slider.valueChanged.connect(self.slider_chaged_)
    # ---------- Thread Initialization-------------------------
        self.opencv_thread = image_thread()
        self.pygame_thread = PyGameMouse_thread()
        self.pygame_IRThread = PyGameIR_thread()
        self.opencv_thread.thesh_value = self.slider.value()

    def slider_chaged_(self):
        self.slider_box.setValue(self.slider.value())
        self.opencv_thread.thesh_value = self.slider.value()

    def load_calib_values(self):
        # Open the file in read mode
        if self.caliberate_load.isChecked():
            with open('caliberation_data.txt', 'r') as file:
                # Iterate over each line in the file
                for i,line in enumerate(file):
                    items = [self.value_tplx,self.value_tply,self.value_tprx , self.value_tpry,self.value_btlx,self.value_btly,self.value_btrx,self.value_btry]
                    items[i].setValue(int(line.strip()))
                    

    def save_calib_values(self):
        if self.caliberate_save.isChecked():
            with open('caliberation_data.txt', 'w') as file:
                # Append some text to the file
                content = f"{self.value_tplx.value()}\n{self.value_tply.value()}\n{self.value_tprx.value()}\n{self.value_tpry.value()}\n{self.value_btlx.value()}\n{self.value_btly.value()}\n{self.value_btrx.value()}\n{self.value_btry.value()} "
                file.write(content)

    def value_add_xy_(self,mssg):
        print(f"im in setvalue {mssg}")
        if (self.caliberate_TPL.isChecked()):
            self.value_tplx.setValue(mssg[0])
            self.value_tply.setValue(mssg[1])
        elif(self.caliberate_TPR.isChecked()):
            self.value_tprx.setValue(mssg[0])
            self.value_tpry.setValue(mssg[1])
        elif(self.caliberate_BTL.isChecked()):
            self.value_btlx.setValue(mssg[0])
            self.value_btly.setValue(mssg[1])
        elif(self.caliberate_BTR.isChecked()):
            self.value_btrx.setValue(mssg[0])
            self.value_btry.setValue(mssg[1])

    def update_xy_(self):
        if (self.caliberate_mode.isChecked()):
                self.opencv_thread.x_y_update.connect(self.value_add_xy_)
        
    def caliberate_load_(self):
        if (self.caliberate_load.isChecked()):
            self.update_output_terminal("Loading caliberation file")
        else:
            
            self.update_output_terminal("Disable caliberation loading")

    def caliberate_mode_(self):
        if (self.caliberate_mode.isChecked()):
            self.opencv_thread.caliberate_on()
            self.update_output_terminal("Caliberation mode activated")
        else:
            self.opencv_thread.caliberate_off()
            self.update_output_terminal("Caliberation mode deactivated")

    def select_bcombo_mode(self):
        self.pygame_thread.select_mode_func(self.comboBox_pygame.currentText())
        print(f"Current selcted moode is : {self.pygame_thread.select_mode}")

    def run_pygame(self):

        if ((self.comboBox.currentText() == "No camera selected") and (self.comboBox_pygame.currentText() == "Mouse")):
            self.update_output_terminal("Starting the pygame --Mouse mode")
            self.pygame_thread.stop_pygame = False
            self.pygame_thread.pygame_end = False
            self.pygame_thread.select_mode_func(self.comboBox_pygame.currentText())
            self.pygame_thread.start()
        elif((self.comboBox.currentText() != "No camera selected")and (self.comboBox_pygame.currentText() == "IR_Pen")):
            if (len(self.image) != 0):
                self.update_output_terminal("Starting the pygame --IR mode")
                self.update_caliberation_coordinates()
                self.pygame_IRThread.stop_pygame = False
                self.pygame_IRThread.pygame_end = False
                self.pygame_IRThread.select_mode_func(self.comboBox_pygame.currentText())
                self.pygame_IRThread.start()
            else:
                self.update_output_terminal("No valid image found")

    def stop_camera_opencv(self):
        if (self.comboBox.currentText() != "No camera selected"):
            self.opencv_thread.stop()
            self.pygame_thread.stop_pygame_functions()
            self.graphicview.scene().clear()
        # self.opencv_thread.stop_signal.emit(True)
        
    def on_radio_button_toggled(self):
        if self.radiobutton.isChecked():
            self.update_output_terminal("Searching for cameras")
            cameras = get_available_cameras()
            self.comboBox.addItems(cameras)
        else:
            self.comboBox.clear()
            self.comboBox.addItem("No camera selected")
            self.update_output_terminal("stop searching cameras")
    
    def run_thread(self):
        current_camera_index = self.window.comboBox.currentIndex()
        camera_name = self.window.comboBox.itemText(current_camera_index)
        self.opencv_thread.camera_index = camera_name
        self.opencv_thread.height = self.graphic_height
        self.opencv_thread.width = self.graphic_width
        self.update_output_terminal("Video Broadcasting")
        self.opencv_thread.message_updated.connect(self.update_output_terminal)
        self.opencv_thread.image_updated.connect(self.render_graphics)
        self.opencv_thread.original_image_update.connect(self.original_image)
        self.opencv_thread.start()

    def update_output_terminal(self,mssg):
        self.textbox.setText(mssg)

    def run_start(self):
        if(self.comboBox.currentText() == "No camera selected"):
            self.update_output_terminal("Please select the camera")
        else:
            self.opencv_thread.not_stoped = False
            self.run_thread()

    def original_image(self,Image):
        self.pygame_IRThread.image = Image

    def render_graphics(self,image):
        self.image = image
        self.pygame_IRThread.image = self.image
        # Convert the numpy array image to a QImage
        # q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_RGB888).rgbSwapped()
        q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_Grayscale8)
        # Get the scene associated with the QGraphicsView
        scene = self.graphicview.scene()
        # If no scene exists, create a new one
        if scene is None:
            scene = QGraphicsScene()
            self.graphicview.setScene(scene)
        else:
            scene.clear()
        # Create a QGraphicsPixmapItem with the QImage
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        pixmap_item.setScale(self.graphicview.width() / pixmap_item.boundingRect().width())
        # Add the QGraphicsPixmapItem to the scene
        scene.addItem(pixmap_item)

    def update_caliberation_coordinates(self):
        self.opencv_thread.value_tplx = int(self.value_tplx.value())
        self.opencv_thread.value_tply = int(self.value_tply.value())
        self.opencv_thread.value_tprx = int(self.value_tprx.value())
        self.opencv_thread.value_tpry = int(self.value_tpry.value())
        self.opencv_thread.value_btlx = int(self.value_btlx.value())
        self.opencv_thread.value_btly = int(self.value_btly.value())
        self.opencv_thread.value_btrx = int(self.value_btrx.value())
        self.opencv_thread.value_btry = int(self.value_btry.value())

if __name__ == "__main__":
    start_capston = capstone()
    start_capston.window.show()
    start_capston.app.exec()