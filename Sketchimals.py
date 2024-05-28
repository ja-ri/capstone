import pygame
from time import sleep
import sys
import numpy as np
from PIL import Image, ImageOps
import cv2
import glob, os
import random
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import pickle
from screeninfo import get_monitors
from main import capstone


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

	def check_for_input(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def change_color(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)
 
def draw_buttons():
    global RED_BUTTON, PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, GREEN_BUTTON, BACK_BUTTON, ANIMALSOUND_BUTTON
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
    BACK_BUTTON = Button(image=None, pos=(50, screen_height-50),
                        text_input="MENU", font=get_font(25), base_color="Black", hovering_color="White")
    for button in [PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, RED_BUTTON, GREEN_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, ANIMALSOUND_BUTTON, BACK_BUTTON]:
        button.change_color(DRAW_MOUSE_POS)
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
    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
    drawing = False
    sound_path = random.choice(os.listdir("Assets/Sounds"))
    str_sound = str(sound_path)
    cut_sound = str_sound.split('.')
    correct_animal = cut_sound[0].lower()
    pygame.mixer.music.stop
    pygame.mixer.music.load("Assets/Sounds/" + sound_path)
    pygame.mixer.music.play(loops=0)
     
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
                if RED_BUTTON.check_for_input(event.pos):
                    color = 'Red'
                    size = 10
               
                elif GREEN_BUTTON.check_for_input(event.pos):
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.check_for_input(event.pos):
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.check_for_input(event.pos):
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.check_for_input(event.pos):
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.check_for_input(event.pos):
                    screen.fill('White')
                    draw_buttons()
                    
                elif BACK_BUTTON.check_for_input(event.pos):
                    main_menu()
                    
                elif ANIMALSOUND_BUTTON.check_for_input(event.pos):
                    pygame.mixer.music.play(loops=0)
                
                    
                elif PREDICT_BUTTON.check_for_input(event.pos):
                    gray_image = process_image()
                    test = crop_image(gray_image)
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('image0.jpg', test)
                    _, test = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)
                    cv2.imwrite('image1.jpg', test)
                    # Define a kernel for morphological operations
                    kernel = np.ones((1, 1), np.uint8)
                    # Apply morphological operations to thin the edges
                    test = cv2.erode(test, kernel, iterations=10)
                    test = test/255.0
                    test = np.expand_dims(test, axis=0)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
                    max_index = np.argmax(prediction)
                    one_hot_encoded = np.zeros_like(prediction)
                    one_hot_encoded[0][max_index] = 1
                    #print(prediction)
                    #print(one_hot_encoded)
                    predicted_variables = encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
                    #print(f"prediction is {predicted_variables}")
                    #print(predicted_variables)
                    #print(correct_animal)
                    max_value = round((prediction.max() * 100), 1)
                    if predicted_variables == correct_animal:
                        predict_text = get_font(25).render(f"You drew a {predicted_variables}", True, "Black", "White") 
                        predict_rect = predict_text.get_rect(center = (screen_width/2 -100, screen_height - 100))
                        screen.blit(predict_text, predict_rect)
                        points_text = get_font(25).render(f"Points: {max_value}/100", True, "Black", "White")
                        points_rect = points_text.get_rect(center = (screen_width/2 -100, screen_height - 50))
                        screen.blit(points_text, points_rect)
                    elif predicted_variables != correct_animal:
                        fail_text = get_font(25).render(f"You were supposed to draw {correct_animal}, Try again!", True, "Black", "White")
                        fail_rect = fail_text.get_rect(center = (screen_width/2 -100, screen_height - 50))
                        screen.blit(fail_text, fail_rect)
                        predict_text = get_font(25).render(f"Prediction is {predicted_variables}", True, "Black", "White") 
                        predict_rect = predict_text.get_rect(center = (screen_width/2 -100, screen_height - 100))
                        screen.blit(predict_text, predict_rect)
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

def draw_gameIR():
        subrect = pygame.Rect(100, 0, screen_width - 100, screen_height)
        sub = screen.subsurface(subrect)
        screen.fill((background_color))
        color = 'Black'
        size = 10
        draw_buttons()
        drawing = False
        last_pos = (0,0)
        (a, s) = pygame.mouse.get_pos()
        
        while True: 
            
            # Find contours
            contours, _ = cv2.findContours(Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate through contours
            for contour in contours:
                # Compute centroid of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    normalized_cX = cX
                    normalized_cY = cY
                    # print(f"image width height {self.image.shape[0],self.image.shape[1]}")
                    # print(f"screen width and height {self.screen_width ,self.screen_height}")
                    # print(f"cx and cy {cX,cY}")
                    # print(f"Normalized cx and cy {normalized_cX,normalized_cY}")
                    end_pos =(normalized_cX,normalized_cY)
                    # pygame.draw.circle(self.screen, color, (normalized_cX, normalized_cY),size)

                    #exclude_x = ((predict_rect.x ),(predict_rect.x + eraser_rect.width))
                    #exclude_y = ((predict_rect.y ),(predict_rect.y + clear_rect.y + clear_rect.height))
                    # print(f"exclude_x[0] {exclude_x[0]}exclude_x[1] {exclude_x[1]}exclude_y[0] {exclude_y[0]}exclude_y[1] {exclude_y[1]}")

                    if ((normalized_cX >= 100)):

                        if ( (last_pos[0] != 0) and (last_pos[1] != 0) ):
                            dx = end_pos[0] - last_pos[0]
                            dy = end_pos[1] - last_pos[1]
                            distance = max(abs(dx), abs(dy))
                            print(f"max_ditstance {distance}")
                            if (distance < 100):
                                for i in range(1, distance + 1):
                                    x = last_pos[0] + int(float(i) / distance * dx)
                                    y = last_pos[1] + int(float(i) / distance * dy)
                                    pygame.draw.circle(screen, color, (x, y), size)
                                last_pos = end_pos  # Update last position
                        last_pos = end_pos
                    else:
                        last_pos = (0,0)

                    if RED_BUTTON.check_for_input(cX,cY):
                        color = 'Red'
                        size = 10
                
                    elif GREEN_BUTTON.check_for_input(cX,cY):
                        color = 'Green'
                        size = 10
                        
                    elif BLUE_BUTTON.check_for_input(cX,cY):
                        color = 'Blue'
                        size = 10
                    
                    elif BLACK_BUTTON.check_for_input(cX,cY):
                        color = 'Black'
                        size = 10
                        
                    elif ERASER_BUTTON.check_for_input(cX,cY):
                        color = 'White'
                        size = 40

                    elif CLEAR_BUTTON.check_for_input(cX,cY):
                        screen.fill('White')
                        draw_buttons()
                        
                    elif PREDICT_BUTTON.check_for_input(cX,cY):
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
                        predicted_variables = encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
                        print(f"prediction is {predicted_variables}")
                        font = pygame.font.Font(None, 60)
                        text_surface = font.render(f"Prediction: {predicted_variables}", True, (0, 0, 0))
                        # text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 20))
                        screen.blit(text_surface,(10,10))    

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
        
        screen.blit(BG1, (0, 0))

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
            button.change_color(OPTIONS_MOUSE_POS)
            button.update(screen)


        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_MOUSE.check_for_input(OPTIONS_MOUSE_POS):
                    draw_game()
                if OPTIONS_TOUCHSCREEN.check_for_input(OPTIONS_MOUSE_POS):
                    draw_game()
                if OPTIONS_IRPEN.check_for_input(OPTIONS_MOUSE_POS):
                    start_capston = capstone()
                    start_capston.window.show()
                    start_capston.app.exec()
                    draw_game()
                if OPTIONS_BACK.check_for_input(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def main_menu():
    while True:
        screen.blit(BG1, (0,0))

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
            button.change_color(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.check_for_input(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.check_for_input(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

pygame.init()
clock = pygame.time.Clock()
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
model = load_model("12_classes.h5")
BG = pygame.image.load("Assets/background1.png")
logo = pygame.image.load("Assets/logo.png")
logo_size = (screen_width/2160, screen_height/1440)
logo1 = pygame.transform.scale(logo, (logo_size))
BG1 = pygame.transform.scale(BG, (screen_width, screen_height))
#code for the background music
pygame.mixer.init()
pygame.mixer.music.load("Assets/music.mp3")
pygame.mixer.music.play(loops=-1)
pygame.mixer.music.set_volume(0.3)

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


#Pig, elephant, cow, frog, monkey, dolphin, parrot, snake, dog, cat, sheep, horse