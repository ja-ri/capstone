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

          
class PyGameMouse_thread(QThread):  

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
        if self.select_mode == "Mouse":
            self.main_ir()
        print("Ending the pygame")
        self.quit()

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
        clear_text = font4.render('Clear self.screen', False, 'White')
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
        output = image[y:y+h, x:x+w]
        dst = cv2.copyMakeBorder(output,int(h/10),int(h/10), int(w/10), int(w/10),cv2.BORDER_CONSTANT, value=0)
        return dst
        

    def draw_game(self):
        subrect = pygame.Rect(100, 0, self.screen_width - 100, self.screen_height)
        sub = self.screen.subsurface(subrect)
        self.screen.fill((self.background_color))
        color = 'Black'
        size = 10
        self.draw_buttons()
        drawing = False

        
        
        while True: 
            for event in pygame.event.get():
                (a, s) = pygame.mouse.get_pos()
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and a>=100:
                    if event.button == 1:  # Left mouse button
                        drawing = True
                        last_pos = pygame.mouse.get_pos()  # Get the starting position
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        drawing = False
                elif event.type == pygame.MOUSEMOTION:
                    if drawing:
                        end_pos = pygame.mouse.get_pos()  # Get the current position
                        pygame.draw.circle(self.screen, color, end_pos, size)  # Draw a circle at the current position

                    # Connect consecutive positions with circles to simulate a line
                        dx = end_pos[0] - last_pos[0]
                        dy = end_pos[1] - last_pos[1]
                        distance = max(abs(dx), abs(dy))
                        for i in range(1, distance + 1):
                            x = last_pos[0] + int(float(i) / distance * dx)
                            y = last_pos[1] + int(float(i) / distance * dy)
                            pygame.draw.circle(self.screen, color, (x, y), size)

                        last_pos = end_pos  # Update last position
                        
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.red_rect.collidepoint(event.pos):
                        color = 'Red'
                        size = 10
                
                    elif self.green_rect.collidepoint(event.pos):
                        color = 'Green'
                        size = 10
                        
                    elif self.blue_rect.collidepoint(event.pos):
                        color = 'Blue'
                        size = 10
                    
                    elif self.black_rect.collidepoint(event.pos):
                        color = 'Black'
                        size = 10
                        
                    elif self.eraser_rect.collidepoint(event.pos):
                        color = 'White'
                        size = 40

                    elif self.clear_rect.collidepoint(event.pos):
                        self.screen.fill('White')
                        self.draw_buttons()
                        
                    elif self.predict_rect.collidepoint(event.pos):
                        gray_image = self.process_image()
                        test = self.crop_image(gray_image)

                        test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                        cv2.imwrite('image0.jpg', test)
                        _, test = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)
                        cv2.imwrite('image1.jpg', test)
                        blurred_image = cv2.GaussianBlur(test, (1, 1), 0)
                        # Define a kernel for morphological operations
                        kernel = np.ones((1, 1), np.uint8)

                        # Apply morphological operations to thin the edges
                        # test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)
                        test = cv2.erode(test, kernel, iterations=10)
                        cv2.imwrite('image2.jpg', test)
                        test = test/255.0
                        
                        
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
                        
                        #testausta
                        #cv2.imwrite('image1.jpg', test)
                        #cv2.imshow("image1.jpg", test)                
                        
                        
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_c:
                        self.screen.fill('White')
                        self.draw_buttons()
            pygame.display.update()




    def main_ir(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        # self.screen_width = GetSystemMetrics(0)
        # self.screen_height = GetSystemMetrics(1)
        self.screen_width = get_monitors()[0].width
        self.screen_height = get_monitors()[0].height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Capstone Project')
        self.background_color = pygame.Color('White')
        self.model = load_model("12_classes.keras")
        self.predict_rect = pygame.Rect(0, 300, 100, 50)
        self.eraser_rect = pygame.Rect(0, 350, 100, 50)
        self.black_rect = pygame.Rect(0, 400, 100, 50)
        self.red_rect = pygame.Rect(0, 450, 100, 50)
        self.green_rect = pygame.Rect(0, 500, 100, 50)
        self.blue_rect = pygame.Rect(0, 550, 100, 50)
        self.clear_rect = pygame.Rect(0, 600, 100, 50)
        self.mytheme = pygame_menu.themes.Theme(background_color =(0, 0, 0, 0), title_background_color = (4, 47, 126), title_font_shadow=True, widget_padding=25)
        self.game_state = "start_menu"
        while True:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if self.game_state == "start_menu":
                self.draw_start_menu()
            if self.game_state == "draw_game":
                self.draw_game()
                
            pygame.display.flip()
