import pygame
from time import sleep
import sys
import pygame_menu
from pygame_menu import themes
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob, os
import random
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import pickle
# from win32api import GetSystemMetrics
import pyautogui
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
import pygame
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtWidgets
import sys
import pickle
from sklearn.preprocessing import OneHotEncoder
import gc
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from keras.utils import to_categorical
from keras.models import load_model
import pickle
import os 
from screeninfo import get_monitors
          
class PyGameIR_thread(QThread):  

    def __init__(self, camera_index=0, parent=None) -> None:
        super().__init__(parent)
        self.image = 0
        self.stop_pygame = False
        self.pygame_end = False
        self.select_mode = ""
        
    def select_mode_func(self,mode):
        self.select_mode = mode

    def stop_pygame_functions(self):
        self.stop_pygame = True

    def run(self):
        print(f"selected thread mode {self.select_mode}")
        print("Starting pygame")
        if self.select_mode == "IR_Pen":
            self.main_ir()
        # pygame.quit()
        print("Ending the pygame")
        # self.quit()

    def draw_buttons(self):
        # default font initialized
        font4 = pygame.font.Font("freesansbold.ttf", 16)
        #Button for predict
        predict_surface = pygame.Surface((100, 50))
        predict_text = font4.render('Predict', False, 'White')
        text1_rect = predict_text.get_rect(center=(predict_surface.get_width()/2, predict_surface.get_height()/2))
        predict_surface.blit(predict_text, text1_rect)
        self.screen.blit(predict_surface, (self.predict_rect.x, self.predict_rect.y))
        
        #Button for switching to eraser
        eraser_surface = pygame.Surface((100, 50))
        eraser_text = font4.render('Eraser', False, 'White')
        text2_rect = eraser_text.get_rect(center=(eraser_surface.get_width()/2, eraser_surface.get_height()/2))
        eraser_surface.blit(eraser_text, text2_rect)
        self.screen.blit(eraser_surface, (self.eraser_rect.x, self.eraser_rect.y))
        
        #Button for switching to black pen
        black_surface = pygame.Surface((100, 50))
        black_text = font4.render('Black pen', False, 'White')
        text6_rect = black_text.get_rect(center=(black_surface.get_width()/2, black_surface.get_height()/2))
        black_surface.blit(black_text, text6_rect)
        self.screen.blit(black_surface, (self.black_rect.x, self.black_rect.y))
        
        #Button for switching to red pen
        red_surface = pygame.Surface((100, 50))
        red_text = font4.render('Red pen', False, 'Red')
        text3_rect = red_text.get_rect(center=(red_surface.get_width()/2, red_surface.get_height()/2))
        red_surface.blit(red_text, text3_rect)
        self.screen.blit(red_surface, (self.red_rect.x, self.red_rect.y))
        
        #Button for switching to green pen
        green_surface = pygame.Surface((100, 50))
        green_text = font4.render('Green pen', False, 'Green')
        text4_rect = green_text.get_rect(center=(green_surface.get_width()/2, green_surface.get_height()/2))
        green_surface.blit(green_text, text4_rect)
        self.screen.blit(green_surface, (self.green_rect.x, self.green_rect.y))
        
        #Button for switching to blue pen
        blue_surface = pygame.Surface((100, 50))
        blue_text = font4.render('Blue pen', False, 'Blue')
        text5_rect = blue_text.get_rect(center=(blue_surface.get_width()/2, blue_surface.get_height()/2))
        blue_surface.blit(blue_text, text5_rect)
        self.screen.blit(blue_surface, (self.blue_rect.x, self.blue_rect.y))
        
        #Button for clearing the whole self.screen
        clear_surface = pygame.Surface((100, 50))
        clear_text = font4.render('Clear screen', False, 'White')
        text7_rect = clear_text.get_rect(center=(clear_surface.get_width()/2, clear_surface.get_height()/2))
        clear_surface.blit(clear_text, text7_rect)
        self.screen.blit(clear_surface, (self.clear_rect.x, self.clear_rect.y))
        
    def draw_start_menu(self):
        #draws start menu
        self.screen.fill((self.background_color))
        mainmenu = pygame_menu.Menu('Capstone', self.screen_width, self.screen_height, theme=self.mytheme)
        mainmenu.add.button('Start', self.draw_game)
        mainmenu.add.button('Quit', pygame_menu.events.EXIT)
        pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
        mainmenu.mainloop(self.screen)
        pygame.display.update()
        
    def process_image(self):
        #processing image to desired format
        subrect = pygame.Rect(100, 0, self.screen_width - 100, self.screen_height)
        sub = self.screen.subsurface(subrect)
        pygame.image.save(sub, 'image1.png')
        image = Image.open("image1.png")
        image = image.save("image1.png")
        image = cv2.imread("image1.png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_not(gray_image)
        return gray_image
        
    def crop_image(self,image):
        #Calculate the bounding rectangle that contains every part of the image
        x, y, w, h = cv2.boundingRect(image)

        #Determine the size of the bounding square
        square_size = max(w, h)

        #Determine the coordinates of the bounding square to center the drawing
        square_x = x + (w - square_size) // 2
        square_y = y + (h - square_size) // 2

        #Check if the square exceeds the dimensions of the image
        if square_x < 0 or square_y < 0 or square_x + square_size > image.shape[1] or square_y + square_size > image.shape[0]:
            # Calculate the amount of extension needed on each side
            left_extension = max(0 - square_x, 0)
            top_extension = max(0 - square_y, 0)
            right_extension = max(square_x + square_size - image.shape[1], 0)
            bottom_extension = max(square_y + square_size - image.shape[0], 0)

            # Extend the image beyond its boundaries
            extended_image = cv2.copyMakeBorder(image, top_extension, bottom_extension, left_extension, right_extension, cv2.BORDER_CONSTANT, value=(0))

            # Adjust the square coordinates due to extension
            square_x += left_extension
            square_y += top_extension

            # The final square x,y,w,h
            square = square_x, square_y, square_size, square_size
            # Final crop
            output = extended_image[square_y:square_y+square_size, square_x:square_x+square_size]
            return output

        else:
            # The final square x,y,w,h
            square = square_x, square_y, square_size, square_size
            # Final crop
            output = image[square_y:square_y+square_size, square_x:square_x+square_size]
            return output
        

    def draw_game(self):
        subrect = pygame.Rect(100, 0, self.screen_width - 100, self.screen_height)
        sub = self.screen.subsurface(subrect)
        self.screen.fill((self.background_color))
        color = 'Black'
        size = 10
        self.draw_buttons()
        drawing = False
        last_pos = (0,0)
        
        while not self.stop_pygame: 
            
            # print(self.screen_height)
            # print(self.screen_width)
            # # Convert frame to grayscale
            # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # # Thresholding to isolate bright areas (IR light)
            # _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate through contours
            for contour in contours:
                # Compute centroid of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Draw centroid on the frame
                    # cv2.circle(self.image, (cX, cY), 5, (0, 0, 255), -1)
                    # Print normalized coordinates
                    # normalized_cX = int(cX * self.screen_width / self.image.shape[0])
                    # normalized_cY = int(cY * self.screen_height / self.image.shape[1])
                    normalized_cX = cX
                    normalized_cY = cY
                    # print(f"image width height {self.image.shape[0],self.image.shape[1]}")
                    # print(f"screen width and height {self.screen_width ,self.screen_height}")
                    # print(f"cx and cy {cX,cY}")
                    # print(f"Normalized cx and cy {normalized_cX,normalized_cY}")
                    end_pos =(normalized_cX,normalized_cY)
                    # pygame.draw.circle(self.screen, color, (normalized_cX, normalized_cY),size)

                    exclude_x = ((self.predict_rect.x ),(self.predict_rect.x + self.eraser_rect.width))
                    exclude_y = ((self.predict_rect.y ),(self.predict_rect.y + self.clear_rect.y + self.clear_rect.height))
                    # print(f"exclude_x[0] {exclude_x[0]}exclude_x[1] {exclude_x[1]}exclude_y[0] {exclude_y[0]}exclude_y[1] {exclude_y[1]}")

                    if ((normalized_cX >= exclude_x[1])):

                        if ( (last_pos[0] != 0) and (last_pos[1] != 0) ):
                            dx = end_pos[0] - last_pos[0]
                            dy = end_pos[1] - last_pos[1]
                            distance = max(abs(dx), abs(dy))
                            print(f"max_ditstance {distance}")
                            if (distance < 100):
                                for i in range(1, distance + 1):
                                    x = last_pos[0] + int(float(i) / distance * dx)
                                    y = last_pos[1] + int(float(i) / distance * dy)
                                    pygame.draw.circle(self.screen, color, (x, y), size)
                                last_pos = end_pos  # Update last position
                        last_pos = end_pos
                    else:
                        last_pos = (0,0)

                    if self.red_rect.collidepoint(normalized_cX,normalized_cY):
                        color = 'Red'
                        size = 10
                
                    elif self.green_rect.collidepoint(normalized_cX,normalized_cY):
                        color = 'Green'
                        size = 10
                        
                    elif self.blue_rect.collidepoint(normalized_cX,normalized_cY):
                        color = 'Blue'
                        size = 10
                    
                    elif self.black_rect.collidepoint(normalized_cX,normalized_cY):
                        color = 'Black'
                        size = 10
                        
                    elif self.eraser_rect.collidepoint(normalized_cX,normalized_cY):
                        color = 'White'
                        size = 40

                    elif self.clear_rect.collidepoint(normalized_cX,normalized_cY):
                        self.screen.fill('White')
                        self.draw_buttons()
                        
                    elif self.predict_rect.collidepoint(normalized_cX,normalized_cY):
                        gray_image = self.process_image()
                        test = self.crop_image(gray_image)
                        cv2.imwrite('image1.jpg', test)
                        test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                        test = cv2.normalize(test, test, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        cv2.imwrite('image2.jpg', test*255)
                        test = np.expand_dims(test, axis=0)
                        with open('encoder.pickle','rb') as f:
                            encode=pickle.load(f)
                        prediction = self.model.predict(test)
                        max_index = np.argmax(prediction)
                        one_hot_encoded = np.zeros_like(prediction)
                        one_hot_encoded[0][max_index] = 1
                        print(prediction)
                        print(one_hot_encoded)
                        print(f"prediction is {encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]}")
                        predicted_variables = encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
                        print(f"prediction is {predicted_variables}")
                        font = pygame.font.Font(None, 60)
                        text_surface = font.render(f"Prediction: {predicted_variables}", True, (0, 0, 0))
                        # text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 20))
                        self.screen.blit(text_surface,(10,10))    

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_pygame  = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.stop_pygame  = True
                    elif event.key == pygame.K_c:
                        self.screen.fill('White')
                        self.draw_buttons()
            
            pygame.display.update()

    def main_ir(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen_width = get_monitors()[0].width

        self.screen_height = get_monitors()[0].height

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Capstone Project')
        self.background_color = pygame.Color('White')
        self.model = load_model("12_classes.h5")
        self.predict_rect = pygame.Rect(0, 300, 100, 50)
        self.eraser_rect = pygame.Rect(0, 350, 100, 50)
        self.black_rect = pygame.Rect(0, 400, 100, 50)
        self.red_rect = pygame.Rect(0, 450, 100, 50)
        self.green_rect = pygame.Rect(0, 500, 100, 50)
        self.blue_rect = pygame.Rect(0, 550, 100, 50)
        self.clear_rect = pygame.Rect(0, 600, 100, 50)
        self.mytheme = pygame_menu.themes.Theme(background_color =(0, 0, 0, 0), title_background_color = (4, 47, 126), title_font_shadow=True, widget_padding=25)
        self.game_state = "start_menu"
        # while not self.stop_pygame:
        self.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_pygame  = True
        if self.game_state == "start_menu":
            self.draw_start_menu()
        if self.game_state == "draw_game":
            self.draw_game()
                
        pygame.display.flip()
