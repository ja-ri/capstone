from traceback import print_tb
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
from main import capstone,main_thread
import threading



def get_font(size): 
    return pygame.font.Font("assets/font.ttf", size)       
            
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
    x, y, w, h = cv2.boundingRect(image)
    output = image[y:y+h,x:x+w]
    top=int(round(h*0.1,0))
    bottom=int(round(h*0.1,0))
    left=int(round(y*0.2,0))
    right=int(round(y*0.2,0))
    print(top)
    padded_image = cv2.copyMakeBorder(
        output,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return padded_image
    

def draw_game():
    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
    drawing = False
    sound_path = random.choice(os.listdir("assets/Sounds"))
    str_sound = str(sound_path)
    cut_sound = str_sound.split('.')
    correct_animal = cut_sound[0].lower()
    pygame.mixer.music.stop
    pygame.mixer.music.load("assets/Sounds/" + sound_path)
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

def get_data_IR():
    with open('caliberation_data.txt', 'r') as file:
        lines = file.readlines()
        print(lines)
        if len(lines) >= 10:  # Ensure there are enough lines
            camera = (int(lines[0].strip("video")))
            thresh = int(lines[1].strip())
            value_tplx = int(lines[2].strip())
            value_tply = int(lines[3].strip())
            value_tprx = int(lines[4].strip())
            value_tpry = int(lines[5].strip())
            value_btlx = int(lines[6].strip())
            value_btly = int(lines[7].strip())
            value_btrx = int(lines[8].strip())
            value_btry = int(lines[9].strip())
            print("Caliberation values loaded successfully")
            return (camera,thresh,value_tplx,value_tply,value_tprx,value_tpry,value_btlx,value_btly,value_btrx,value_btry)
        else:
            print("Insufficient data in the file")
            return 0

def draw_gameIR():

    screen.fill((background_color))
    color = 'Black'
    size = 10
    draw_buttons()
    drawing = False
    sound_path = random.choice(os.listdir("assets/Sounds"))
    str_sound = str(sound_path)
    cut_sound = str_sound.split('.')
    correct_animal = cut_sound[0].lower()
    pygame.mixer.music.stop
    pygame.mixer.music.load("assets/Sounds/" + sound_path)
    pygame.mixer.music.play(loops=0)
    last_pos = (0,0)

    caliberation_data = get_data_IR()
    print(caliberation_data)
    cap = cv2.VideoCapture(caliberation_data[0])  # Open the selected camera
    
    if not cap.isOpened():
        print()("Error: Failed to open camera.")
        return

    while True: 

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

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        height, width = frame.shape[:2]

        # (camera,thresh,value_tplx,value_tply,value_tprx,value_tpry,value_btlx,value_btly,value_btrx,value_btry)
        # Calculate the slice dimensions
        start_x = min(caliberation_data[2], width)
        start_y = min(caliberation_data[3], height)
        end_x = min(caliberation_data[4], width)
        end_y = min(caliberation_data[7], height)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to isolate bright areas (IR light)
        _, thresh = cv2.threshold(gray, caliberation_data[1], 255, cv2.THRESH_BINARY)

        cropped_image = thresh[start_y:end_y, start_x:end_x]
        cropped_image = cv2.resize(cropped_image,(get_monitors()[0].width ,get_monitors()[0].height))
        
        # Find contours
        contours, _ = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through contours
        for contour in contours:
            # Compute centroid of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                end_pos = (cX, cY)

                if RED_BUTTON.check_for_input(end_pos):
                    color = 'Red'
                    size = 10
               
                elif GREEN_BUTTON.check_for_input(end_pos):
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.check_for_input(end_pos):
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.check_for_input(end_pos):
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.check_for_input(end_pos):
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.check_for_input(end_pos):
                    screen.fill('White')
                    draw_buttons()
                    
                elif BACK_BUTTON.check_for_input(end_pos):
                    main_menu()
                    
                elif ANIMALSOUND_BUTTON.check_for_input(end_pos):
                    pygame.mixer.music.play(loops=0)
                
                    
                elif PREDICT_BUTTON.check_for_input(end_pos):
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
                else:
                    
                    if last_pos != (0, 0):
                        dx = end_pos[0] - last_pos[0]
                        dy = end_pos[1] - last_pos[1]
                        distance = max(abs(dx), abs(dy))
                        if distance < 50:
                            for i in range(1, distance + 1):
                                x = last_pos[0] + int(float(i) / distance * dx)
                                y = last_pos[1] + int(float(i) / distance * dy)
                                pygame.draw.circle(screen, color, (x, y), size)
                            last_pos = end_pos  # Update last position
                    last_pos = end_pos
            else:
                last_pos = (0, 0)

        pygame.display.update()
        pygame.time.wait(10)  # Add a small delay to avoid high CPU usage        # for event in pygame.event.get():
     
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
                    pyqt_thread = threading.Thread(target=main_thread)
                    # # pyqt_thread.start()
                    # # main_thread()
                    # pyqt_thread = threading.Thread(target=draw_gameIR)
                    # pyqt_thread.start()
                    # pyqt_thread.join() 
                    draw_gameIR()
                    print("main done")
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
BG = pygame.image.load("assets/background1.png")
logo = pygame.image.load("assets/logo.png")
logo_size = (screen_width/2160, screen_height/1440)
logo1 = pygame.transform.scale(logo, (logo_size))
BG1 = pygame.transform.scale(BG, (screen_width, screen_height))
#code for the background music
pygame.mixer.init()
pygame.mixer.music.load("assets/music.mp3")
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