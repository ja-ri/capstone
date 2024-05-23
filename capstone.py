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
from win32api import GetSystemMetrics

def get_font(size): 
    return pygame.font.Font("Assets/font.ttf", size)       
            
class Button():
	def __init__(self, image, pos, text_input, font, base_color, hovering_color):
		self.image = image
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		self.font = font
		self.base_color, self.hovering_color = base_color, hovering_color
		self.text_input = text_input
		self.text = self.font.render(self.text_input, True, self.base_color)
		if self.image is None:
		    self.image = self.text
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos))

	def update(self, screen):
		if self.image is not None:
			screen.blit(self.image, self.rect)
		screen.blit(self.text, self.text_rect)

	def checkForInput(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def changeColor(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)
 
def draw_buttons():
    global RED_BUTTON, PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, GREEN_BUTTON, BACK_BUTTON, MUSIC_BUTTON, ANIMALSOUND_BUTTON
    DRAW_MOUSE_POS = pygame.mouse.get_pos()
    PREDICT_BUTTON = Button(image=None, pos=(50, 300), 
                        text_input="PREDICT", font=get_font(25), base_color="Black", hovering_color="White")
    ERASER_BUTTON = Button(image=None, pos=(50, 350), 
                        text_input="ERASER", font=get_font(25), base_color="Black", hovering_color="White")
    BLACK_BUTTON = Button(image=None, pos=(50, 400), 
                        text_input="BLACK", font=get_font(25), base_color="Black", hovering_color="White")
    RED_BUTTON = Button(image=None, pos=(50, 450), 
                        text_input="RED", font=get_font(25), base_color="Red", hovering_color="White")
    GREEN_BUTTON = Button(image=None, pos=(50, 500), 
                        text_input="GREEN", font=get_font(25), base_color="Green", hovering_color="White")
    BLUE_BUTTON = Button(image=None, pos=(50, 550), 
                        text_input="BLUE", font=get_font(25), base_color="Blue", hovering_color="White")
    CLEAR_BUTTON = Button(image=None, pos=(50, 600), 
                        text_input="CLEAR", font=get_font(25), base_color="Black", hovering_color="White")
    ANIMALSOUND_BUTTON = Button(image=None, pos=(50, 50),
                        text_input="SOUND", font=get_font(25), base_color="Black", hovering_color="White")
    MUSIC_BUTTON = Button(image=None, pos=(50, 100),
                        text_input="MUSIC", font=get_font(25), base_color="Black", hovering_color="White")
    BACK_BUTTON = Button(image=None, pos=(50, screen_height-50),
                        text_input="MENU", font=get_font(25), base_color="Black", hovering_color="White")
    for button in [PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, RED_BUTTON, GREEN_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, ANIMALSOUND_BUTTON, MUSIC_BUTTON, BACK_BUTTON]:
        button.changeColor(DRAW_MOUSE_POS)
        button.update(screen)

    
def process_image():
    #processing image to desired format
    subrect = pygame.Rect(100, 0, screen_width - 100, screen_height)
    sub = screen.subsurface(subrect)
    pygame.image.save(sub, 'image1.png')
    image = Image.open("image1.png")
    image = image.save("image1.png")
    image = cv2.imread("image1.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)
    return gray_image
    
def crop_image(image):
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
    

def draw_game():
    subrect = pygame.Rect(100, 0, screen_width - 100, screen_height)
    sub = screen.subsurface(subrect)
    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
    drawing = False
    DRAW_MOUSE_POS = pygame.mouse.get_pos()
    
    
    
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
                    pygame.draw.circle(screen, color, end_pos, size)  # Draw a circle at the current position

                # Connect consecutive positions with circles to simulate a line
                    dx = end_pos[0] - last_pos[0]
                    dy = end_pos[1] - last_pos[1]
                    distance = max(abs(dx), abs(dy))
                    for i in range(1, distance + 1):
                        x = last_pos[0] + int(float(i) / distance * dx)
                        y = last_pos[1] + int(float(i) / distance * dy)
                        pygame.draw.circle(screen, color, (x, y), size)

                    last_pos = end_pos  # Update last position
                    
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if RED_BUTTON.checkForInput(event.pos):
                    color = 'Red'
                    size = 10
               
                elif GREEN_BUTTON.checkForInput(event.pos):
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.checkForInput(event.pos):
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.checkForInput(event.pos):
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.checkForInput(event.pos):
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.checkForInput(event.pos):
                    screen.fill('White')
                    draw_buttons()
                    
                elif BACK_BUTTON.checkForInput(event.pos):
                    main_menu()
                    
                elif PREDICT_BUTTON.checkForInput(event.pos):
                    gray_image = process_image()
                    test = crop_image(gray_image)
                    cv2.imwrite('image1.jpg', test)
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                    test = cv2.normalize(test, test, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    
                    cv2.imwrite('image2.jpg', test*255)
                    test = np.expand_dims(test, axis=0)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
                    max_index = np.argmax(prediction)
                    one_hot_encoded = np.zeros_like(prediction)
                    one_hot_encoded[0][max_index] = 1
                    max_value = round((prediction.max() * 100), 1)
                    predict_text = get_font(25).render(f"Prediction is {encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]}", True, "Black", "White") 
                    predict_rect = predict_text.get_rect(center = (screen_width/2 -100, screen_height - 100))
                    screen.blit(predict_text, predict_rect)
                    points_text = get_font(25).render(f"Points: {max_value}/100", True, "Black", "White")
                    points_rect = points_text.get_rect(center = (screen_width/2 -100, screen_height - 50))
                    screen.blit(points_text, points_rect)
                    pygame.display.update()              
                    
                    

            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_c:
                    screen.fill('White')
                    draw_buttons()
        pygame.display.update()

def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()
        
        screen.blit(BG, (0, 0))

        OPTIONS_TEXT = get_font(75).render("Options", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(screen_width/2, 100))
        screen.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_MOUSE = Button(image=None, pos=(screen_width/2, 315), 
                            text_input="MOUSE", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_TOUCHSCREEN = Button(image=None, pos=(screen_width/2, 465), 
                            text_input="TOUCHSCREEN", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_IRPEN = Button(image=None, pos=(screen_width/2, 615), 
                            text_input="IR PEN", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_BACK = Button(image=None, pos=(screen_width/2, 765), 
                            text_input="BACK", font=get_font(100), base_color="Black", hovering_color="White")

        for button in [OPTIONS_MOUSE, OPTIONS_TOUCHSCREEN, OPTIONS_IRPEN, OPTIONS_BACK]:
            button.changeColor(OPTIONS_MOUSE_POS)
            button.update(screen)


        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_MOUSE.checkForInput(OPTIONS_MOUSE_POS):
                    draw_game()
                #if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                 #   options()
                if OPTIONS_TOUCHSCREEN.checkForInput(OPTIONS_MOUSE_POS):
                    draw_game()
                if OPTIONS_IRPEN.checkForInput(OPTIONS_MOUSE_POS):
                    draw_game()
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def main_menu():
    while True:
        screen.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()
        
        #logo_rect = logo1.get_rect()
        #logo_rect.center = (screen_width / 2, screen_height /6)
        #screen.blit(logo1, logo_rect.topleft)
        MENU_TEXT = get_font(150).render("SKETCHIMALS", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(screen_width/2, 120))

        PLAY_BUTTON = Button(image=None, pos=(screen_width/2, 390), 
                            text_input="PLAY", font=get_font(100), base_color="#000000", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(screen_width/2, 540), 
                            text_input="QUIT", font=get_font(100), base_color="#000000", hovering_color="White")

        screen.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

pygame.init()
clock = pygame.time.Clock()
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
model = load_model("12_classes.h5")
BG = pygame.image.load("Assets/background1.png")
logo = pygame.image.load("Assets/logo.png")
logo_size = (screen_width/2160, screen_height/1440)
logo1 = pygame.transform.scale(logo, (logo_size))


predict_rect = pygame.Rect(screen_width/2 -100, screen_height - 100, 400, 50)
points_rect = pygame.Rect(screen_width/2 - 100, screen_height - 50, 400, 50)

game_state = "start_menu"


while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if game_state == "start_menu":
        main_menu()
    if game_state == "draw_game":
        draw_game()
        
    pygame.display.flip()
