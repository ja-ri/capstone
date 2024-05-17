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

<<<<<<< Updated upstream
          
            
def draw_buttons():
=======
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
       
def get_font(size): 
    return pygame.font.Font("Assets/font.ttf", size)            
            
##def draw_buttons():
>>>>>>> Stashed changes
    # default font initialized
    font4 = pygame.font.Font("freesansbold.ttf", 16)
    #Button for predict
    predict_surface = pygame.Surface((100, 50))
    predict_text = font4.render('Predict', False, 'White')
    text1_rect = predict_text.get_rect(center=(predict_surface.get_width()/2, predict_surface.get_height()/2))
    predict_surface.blit(predict_text, text1_rect)
    screen.blit(predict_surface, (predict_rect.x, predict_rect.y))
    
    #Button for switching to eraser
    eraser_surface = pygame.Surface((100, 50))
    eraser_text = font4.render('Eraser', False, 'White')
    text2_rect = eraser_text.get_rect(center=(eraser_surface.get_width()/2, eraser_surface.get_height()/2))
    eraser_surface.blit(eraser_text, text2_rect)
    screen.blit(eraser_surface, (eraser_rect.x, eraser_rect.y))
    
    #Button for switching to black pen
    black_surface = pygame.Surface((100, 50))
    black_text = font4.render('Black pen', False, 'White')
    text6_rect = black_text.get_rect(center=(black_surface.get_width()/2, black_surface.get_height()/2))
    black_surface.blit(black_text, text6_rect)
    screen.blit(black_surface, (black_rect.x, black_rect.y))
    
    #Button for switching to red pen
    red_surface = pygame.Surface((100, 50))
    red_text = font4.render('Red pen', False, 'Red')
    text3_rect = red_text.get_rect(center=(red_surface.get_width()/2, red_surface.get_height()/2))
    red_surface.blit(red_text, text3_rect)
    screen.blit(red_surface, (red_rect.x, red_rect.y))
    
    #Button for switching to green pen
    green_surface = pygame.Surface((100, 50))
    green_text = font4.render('Green pen', False, 'Green')
    text4_rect = green_text.get_rect(center=(green_surface.get_width()/2, green_surface.get_height()/2))
    green_surface.blit(green_text, text4_rect)
    screen.blit(green_surface, (green_rect.x, green_rect.y))
    
    #Button for switching to blue pen
    blue_surface = pygame.Surface((100, 50))
    blue_text = font4.render('Blue pen', False, 'Blue')
    text5_rect = blue_text.get_rect(center=(blue_surface.get_width()/2, blue_surface.get_height()/2))
    blue_surface.blit(blue_text, text5_rect)
    screen.blit(blue_surface, (blue_rect.x, blue_rect.y))
    
    #Button for clearing the whole screen
    clear_surface = pygame.Surface((100, 50))
    clear_text = font4.render('Clear screen', False, 'White')
    text7_rect = clear_text.get_rect(center=(clear_surface.get_width()/2, clear_surface.get_height()/2))
    clear_surface.blit(clear_text, text7_rect)
    screen.blit(clear_surface, (clear_rect.x, clear_rect.y))


<<<<<<< Updated upstream
mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0), title_background_color = (4, 47, 126), title_font_shadow=True, widget_padding=25)
  

def draw_start_menu():
    #draws start menu
    screen.fill((background_color))
    mainmenu = pygame_menu.Menu('Capstone', screen_width, screen_height, theme=mytheme)
    mainmenu.add.button('Start', draw_game)
    mainmenu.add.button('Quit', pygame_menu.events.EXIT)
    pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
  
    mainmenu.mainloop(screen)
    pygame.display.update()
=======
#mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0), title_background_color = (4, 47, 126), title_font_shadow=True, widget_padding=25)
  


##def draw_start_menu():
    #draws start menu
   # screen.fill((background_color))
    #mainmenu = pygame_menu.Menu('Capstone', screen_width, screen_height, theme=mytheme)
    #mainmenu.add.button('Start', draw_game)
    #mainmenu.add.button('Quit', pygame_menu.events.EXIT)
    #pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
  
    #mainmenu.mainloop(screen)
    #pygame.display.update()
>>>>>>> Stashed changes
    
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
    
<<<<<<< Updated upstream

def draw_game():
    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
=======
def draw_game():
    screen.fill(background_color)
    color = 'Black'
    size = 10
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
    
    for button in [PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, RED_BUTTON, GREEN_BUTTON, BLUE_BUTTON, CLEAR_BUTTON]:
        button.changeColor(DRAW_MOUSE_POS)
        button.update(screen)
>>>>>>> Stashed changes
    
    while True: 
        for event in pygame.event.get():
            (a, s) = pygame.mouse.get_pos() 
            if event.type == pygame.MOUSEMOTION and a >= 100:
                if event.buttons[0]:  
                    last = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    pygame.draw.line(screen, color, last, event.pos, size)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
<<<<<<< Updated upstream
                if red_rect.collidepoint(event.pos):
                    color = 'Red'
                    size = 10
               
                elif green_rect.collidepoint(event.pos):
                    color = 'Green'
                    size = 10
                    
                elif blue_rect.collidepoint(event.pos):
                    color = 'Blue'
                    size = 10
                
                elif black_rect.collidepoint(event.pos):
                    color = 'Black'
                    size = 10
                    
                elif eraser_rect.collidepoint(event.pos):
                    color = 'White'
                    size = 40

                elif clear_rect.collidepoint(event.pos):
                    screen.fill('White')
                    draw_buttons()
                    
                elif predict_rect.collidepoint(event.pos):
=======
                if RED_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    color = 'Red'
                    size = 10
               
                elif GREEN_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.checkForInput(DRAW_MOUSE_POS):
                    screen.fill('White')
                    draw_buttons()
                    
                elif PREDICT_BUTTON.checkForInput(DRAW_MOUSE_POS):
>>>>>>> Stashed changes
                    gray_image = process_image()
                    test = crop_image(gray_image)
                    cv2.imwrite('image1.jpg', test)
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite('image2.jpg', test)
                    test = np.expand_dims(test, axis=0)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
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
                    screen.fill('White')
                    draw_buttons()
        pygame.display.update()
<<<<<<< Updated upstream


=======
##def draw_game():
    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
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
                if red_rect.collidepoint(event.pos):
                    color = 'Red'
                    size = 10
               
                elif green_rect.collidepoint(event.pos):
                    color = 'Green'
                    size = 10
                    
                elif blue_rect.collidepoint(event.pos):
                    color = 'Blue'
                    size = 10
                
                elif black_rect.collidepoint(event.pos):
                    color = 'Black'
                    size = 10
                    
                elif eraser_rect.collidepoint(event.pos):
                    color = 'White'
                    size = 40

                elif clear_rect.collidepoint(event.pos):
                    screen.fill('White')
                    draw_buttons()
                    
                elif predict_rect.collidepoint(event.pos):
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
                    screen.fill('White')
                    draw_buttons()
        pygame.display.update()

def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()
        
        screen.blit(BG, (0, 0))

        OPTIONS_TEXT = get_font(75).render("Options", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(960, 100))
        screen.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_MOUSE = Button(image=None, pos=(960, 315), 
                            text_input="MOUSE", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_TOUCHSCREEN = Button(image=None, pos=(960, 465), 
                            text_input="TOUCHSCREEN", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_ARPEN = Button(image=None, pos=(960, 615), 
                            text_input="AR PEN", font=get_font(100), base_color="Black", hovering_color="White")
        OPTIONS_BACK = Button(image=None, pos=(960, 765), 
                            text_input="BACK", font=get_font(100), base_color="Black", hovering_color="White")

        for button in [OPTIONS_MOUSE, OPTIONS_TOUCHSCREEN, OPTIONS_ARPEN, OPTIONS_BACK]:
            button.changeColor(OPTIONS_MOUSE_POS)
            button.update(screen)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def main_menu():
    while True:
        screen.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(150).render("SKETCHIMALS", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(960, 120))

        PLAY_BUTTON = Button(image=None, pos=(960, 390), 
                            text_input="PLAY", font=get_font(100), base_color="#000000", hovering_color="White")
        OPTIONS_BUTTON = Button(image=None, pos=(960, 540), 
                            text_input="OPTIONS", font=get_font(100), base_color="#000000", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(960, 690), 
                            text_input="QUIT", font=get_font(100), base_color="#000000", hovering_color="White")

        screen.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    draw_game()
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
>>>>>>> Stashed changes
pygame.init()
clock = pygame.time.Clock()
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
model = load_model("12_classes.h5")
<<<<<<< Updated upstream
=======
BG = pygame.image.load("Assets/Background.png")
>>>>>>> Stashed changes


predict_rect = pygame.Rect(0, 300, 100, 50)
eraser_rect = pygame.Rect(0, 350, 100, 50)
black_rect = pygame.Rect(0, 400, 100, 50)
red_rect = pygame.Rect(0, 450, 100, 50)
green_rect = pygame.Rect(0, 500, 100, 50)
blue_rect = pygame.Rect(0, 550, 100, 50)
clear_rect = pygame.Rect(0, 600, 100, 50)
game_state = "start_menu"


while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if game_state == "start_menu":
<<<<<<< Updated upstream
        draw_start_menu()
=======
        main_menu()
>>>>>>> Stashed changes
    if game_state == "draw_game":
        draw_game()
        
    pygame.display.flip()
