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
        q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_RGB888).rgbSwapped()
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