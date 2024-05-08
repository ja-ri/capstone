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

          
            
def draw_buttons():

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


mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0), title_background_color = (4, 47, 126), title_font_shadow=True, widget_padding=25)
  

def draw_start_menu():
    screen.fill((background_color))
    mainmenu = pygame_menu.Menu('Capstone', screen_width, screen_height, theme=mytheme)
    mainmenu.add.button('Start', draw_game)
    mainmenu.add.button('Quit', pygame_menu.events.EXIT)
    pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
  
    mainmenu.mainloop(screen)
    pygame.display.update()
    
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
    
    while True: 
        for event in pygame.event.get():
            (a, s) = pygame.mouse.get_pos() 
            if event.type == pygame.MOUSEMOTION and a >= 100:
                if event.buttons[0]:  
                    last = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    pygame.draw.line(screen, color, last, event.pos, size)
                    
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
                    pygame.image.save(sub, 'image1.png')
                    image = Image.open("image1.png")
                    image = image.save("image1.png")
                    image = cv2.imread("image1.png")
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #cv2.imwrite('image1.jpg', gray_image)
                    gray_image = cv2.bitwise_not(gray_image)
                    #print(gray_image.shape)
                    #x,y,w,h = cv2.boundingRect(gray_image) #leveempi sivu kokonaan, lyhyemmän sivu puoleenväliin ja siitä molempiin suuntiin puolikas pidempää sivua
                    #gray_image = gray_image[y:h+y, x:w+x]
                    #cv2.imwrite = cv2('image1.jpg', gray_image)
                    #plt.imshow(test)
                    test = crop_image(gray_image)
                    #cv2.imwrite('image1.jpg', test)
                    #cv2.imshow("image1.jpg", test)
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                    #cv2.imwrite('image2.jpg', test)
                    #cv2.imshow("image2.jpg", test)
                    test = np.expand_dims(test, axis=0)
                    #print(test.shape)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
                    #encode_names = np.array([""]).reshape(-1,1)
                    #encode = OneHotEncoder(handle_unknown='error')
                    #encoded_names = encode.fit_transform(encode_names).toarray()
                    max_index = np.argmax(prediction)
                    one_hot_encoded = np.zeros_like(prediction)
                    one_hot_encoded[0][max_index] = 1
                    print(prediction)
                    print(one_hot_encoded)
                    print(f"prediction is {encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]}")
                    
                    #testausta
                    #cv2.imwrite('image1.jpg', test)
                    #cv2.imshow("image1.jpg", test)
                    #print(x,y,w,h)
                    #print(gray_image.shape)
                    #crop_image(image)
                    
                    
                    
                    

            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
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

#btn_rect = pygame.Rect(600, 600, 100, 100)
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
        draw_start_menu()
    if game_state == "draw_game":
        draw_game()
        
    pygame.display.flip()
