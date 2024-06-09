import pygame
import sys
import numpy as np
from PIL import Image
import cv2
import glob, os
import random
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import pickle
from screeninfo import get_monitors
from main import main_thread
import threading      
            
class Button():     #defines buttons class
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

	def update(self, screen):   #adds buttons to the screen
		if self.image is not None:
			screen.blit(self.image, self.rect)
		screen.blit(self.text, self.text_rect)

	def check_for_input(self, position):    #checks for user input
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def change_color(self, position):   #changes buttons color when mouse hovers over them
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)
 
def get_font(size): #sets font
    return pygame.font.Font("assets/font.ttf", size)  

def draw_buttons(): #defines and draws all the buttons
    global RED_BUTTON, PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, GREEN_BUTTON, BACK_BUTTON, ANIMALSOUND_BUTTON, NEXT_BUTTON
    DRAW_MOUSE_POS = pygame.mouse.get_pos()
    PREDICT_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 - screen_height/30 * 3), 
                        text_input="PREDICT", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    ERASER_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 - screen_height/30 * 2), 
                        text_input="ERASER", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    BLACK_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 - screen_height/30), 
                        text_input="BLACK", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    RED_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2), 
                        text_input="RED", font=get_font(font_size), base_color="Red", hovering_color="#D9DDDC")
    GREEN_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 + screen_height/30), 
                        text_input="GREEN", font=get_font(font_size), base_color="Green", hovering_color="#D9DDDC")
    BLUE_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 + screen_height/30 * 2), 
                        text_input="BLUE", font=get_font(font_size), base_color="Blue", hovering_color="#D9DDDC")
    CLEAR_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/2 + screen_height/30 * 3), 
                        text_input="CLEAR", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    ANIMALSOUND_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/20),
                        text_input="SOUND", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    NEXT_BUTTON = Button(image=None, pos=(screen_width/20, screen_height/20 + screen_height/30),
                        text_input="NEXT", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    BACK_BUTTON = Button(image=None, pos=(screen_width/20, screen_height - screen_height/20),
                        text_input="MENU", font=get_font(font_size), base_color="Black", hovering_color="#D9DDDC")
    for button in [PREDICT_BUTTON, ERASER_BUTTON, BLACK_BUTTON, RED_BUTTON, GREEN_BUTTON, BLUE_BUTTON, CLEAR_BUTTON, ANIMALSOUND_BUTTON, BACK_BUTTON, NEXT_BUTTON]:
        button.change_color(DRAW_MOUSE_POS)
        button.update(screen)
        
    
def process_image():
    #processing image to desired format and returns it
    subrect = pygame.Rect(screen_width/10, 0, screen_width - screen_width/10, screen_height)
    sub = screen.subsurface(subrect)
    pygame.image.save(sub, 'image1.png')
    image = Image.open("image1.png")
    image = image.save("image1.png")
    image = cv2.imread("image1.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(gray_image)
    return gray_image
    
def crop_image(image):  #crops the image to desired format (square) and returns it
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
    
def sound_to_str(sound):    #takes the name of the sound file and converts it to a string
    str_sound = str(sound)
    cut_sound = str_sound.split('.')
    correct_animal = cut_sound[0].lower()
    return correct_animal

def draw_game():
    screen.fill((background_color))
    color = 'Black' #pen color
    size = 10       #pen size
    drawing = False
    sound_path = random.choice(os.listdir("assets/Sounds")) #gives random animal sound
    pygame.mixer.music.stop #stops the menu music
    pygame.mixer.music.load("assets/Sounds/" + sound_path)  #loads the animal sound
    pygame.mixer.music.play(loops=0)    #plays the animal sound once
     
    while True: 
        draw_buttons()#draws the buttons and adds the color change when hovering over them
        for event in pygame.event.get():
            (a, s) = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and a>=screen_width/10:#limits the drawing screen a bit so that you can't draw over the buttons
                if event.button == 1:  # Left mouse button
                    drawing = True  #starts drawing when left mousebutton is pressed
                    last_pos = pygame.mouse.get_pos()  # Get the starting position
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False #ends drawing when left mouse button is released
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
                if RED_BUTTON.check_for_input(event.pos):   #changes the pen color to red
                    color = 'Red'
                    size = 10
            
                elif GREEN_BUTTON.check_for_input(event.pos):   #changes the pen color to green
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.check_for_input(event.pos):   #changes the pen color to blue
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.check_for_input(event.pos):   #changes the pen color to black
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.check_for_input(event.pos):     #changes the pen color to white and increases the size to mimic eraser
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.check_for_input(event.pos):   #clears the screen if player wants to start over the whole drawing
                    screen.fill('White')
                    draw_buttons()
                    
                elif BACK_BUTTON.check_for_input(event.pos):    #exits to main menu
                    main_menu()
                    
                elif ANIMALSOUND_BUTTON.check_for_input(event.pos): #plays the current animal sound again
                    pygame.mixer.music.play(loops=0)

                elif NEXT_BUTTON.check_for_input(event.pos):    #gives new animal to draw
                        draw_game()
                        
                elif PREDICT_BUTTON.check_for_input(event.pos): #when drawing is finished gives the image to AI model and it predicts what animal was drawn and gives score based on it's accuracy
                    gray_image = process_image()    #processing image to desired format (grayscale) and returns it
                    test = crop_image(gray_image)   #crops the image to desired format (square) and returns it
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)  #resizes the image to 28x28 pixels so the model can use it
                    cv2.imwrite('image0.jpg', test)
                    _, test = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)
                    cv2.imwrite('image1.jpg', test)
                    kernel = np.ones((1, 1), np.uint8)  # Define a kernel for morphological operations
                    test = cv2.erode(test, kernel, iterations=10)   # Apply morphological operations to thin the edges
                    test = test/255.0
                    test = np.expand_dims(test, axis=0)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
                    max_index = np.argmax(prediction)
                    one_hot_encoded = np.zeros_like(prediction)
                    one_hot_encoded[0][max_index] = 1
                    predicted_variables = encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
                    max_value = round((prediction.max() * 100), 1)  #calculates the points for player
                    if predicted_variables == sound_to_str(sound_path): #if the prediction matches the sound
                        predict_text = get_font(font_size).render(f"You drew a {predicted_variables}", True, "Black", "White") 
                        predict_rect = predict_text.get_rect(center = (screen_width/2, screen_height - screen_height/20))
                        screen.blit(predict_text, predict_rect)
                        points_text = get_font(font_size).render(f"Points: {max_value}/100", True, "Black", "White")
                        points_rect = points_text.get_rect(center = (screen_width/2, screen_height - screen_height/10))
                        screen.blit(points_text, points_rect)
                    elif predicted_variables != sound_to_str(sound_path):   #if the prediction doesn't match the sound
                        fail_text = get_font(font_size).render(f"You were supposed to draw {sound_to_str(sound_path)}, Try again!", True, "Black", "White")
                        fail_rect = fail_text.get_rect(center = (screen_width/2, screen_height - screen_height/10))
                        screen.blit(fail_text, fail_rect)
                        predict_text = get_font(font_size).render(f"Prediction is {predicted_variables}", True, "Black", "White") 
                        predict_rect = predict_text.get_rect(center = (screen_width/2, screen_height - screen_height/20))
                        screen.blit(predict_text, predict_rect)
                    pygame.display.update()              
                
                

            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:    #exits the game if "esc" is pressed
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_c: #clears the screen if key "c" is pressed
                    screen.fill('White')
                    draw_buttons()
            pygame.display.update()

def get_data_IR():
    with open('calibration_data.txt', 'r') as file:
        lines = file.readlines()
        print(lines)
        if len(lines) >= 10:  # Ensure there are enough lines
            camera = (int(lines[0].strip("video")))
            thresh = int(lines[1])
            value_tplx = int(lines[2])
            value_tply = int(lines[3])
            value_tprx = int(lines[4])
            value_tpry = int(lines[5])
            value_btlx = int(lines[6])
            value_btly = int(lines[7])
            value_btrx = int(lines[8])
            value_btry = int(lines[9])
            print("Calibration values loaded successfully")
            return (camera,thresh,value_tplx,value_tply,value_tprx,value_tpry,value_btlx,value_btly,value_btrx,value_btry)
        else:
            print("Insufficient data in the file")
            return 0

def draw_gameIR():
    screen.fill((background_color))
    color = 'Black' #pen color
    size = 10       #pen size
    sound_path = random.choice(os.listdir("assets/Sounds")) #gives random animal sound
    pygame.mixer.music.stop #stops the menu music
    pygame.mixer.music.load("assets/Sounds/" + sound_path)  #loads the animal sound
    pygame.mixer.music.play(loops=0)    #plays the animal sound once
    last_pos = (0,0)

    calibration_data = get_data_IR()
    print(calibration_data)
    cap = cv2.VideoCapture(calibration_data[0])  # Open the selected camera
    
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    while True: 
        draw_buttons()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:    #exits the game if "esc" is pressed
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_c:   #clears the screen if key "c" is pressed
                    screen.fill('White')
                    draw_buttons()

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        height, width = frame.shape[:2]

        # (camera,thresh,value_tplx,value_tply,value_tprx,value_tpry,value_btlx,value_btly,value_btrx,value_btry)
        # Calculate the slice dimensions
        start_x = min(calibration_data[2], width)
        start_y = min(calibration_data[3], height)
        end_x = min(calibration_data[4], width)
        end_y = min(calibration_data[7], height)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to isolate bright areas (IR light)
        _, thresh = cv2.threshold(gray, calibration_data[1], 255, cv2.THRESH_BINARY)

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

                if RED_BUTTON.check_for_input(end_pos):   #changes the pen color to red
                    color = 'Red'
                    size = 10
            
                elif GREEN_BUTTON.check_for_input(end_pos):   #changes the pen color to green
                    color = 'Green'
                    size = 10
                    
                elif BLUE_BUTTON.check_for_input(end_pos):   #changes the pen color to blue
                    color = 'Blue'
                    size = 10
                
                elif BLACK_BUTTON.check_for_input(end_pos):   #changes the pen color to black
                    color = 'Black'
                    size = 10
                    
                elif ERASER_BUTTON.check_for_input(end_pos):    #changes the pen color to white and increases the size to mimic eraser
                    color = 'White'
                    size = 40

                elif CLEAR_BUTTON.check_for_input(end_pos):     #clears the screen if player wants to start over the whole drawing
                    screen.fill('White')
                    draw_buttons()
                    
                elif BACK_BUTTON.check_for_input(end_pos):  #exits to main menu
                    main_menu()
                    
                elif ANIMALSOUND_BUTTON.check_for_input(end_pos):   #plays the animal sound again
                    pygame.mixer.music.play(loops=0)

                elif NEXT_BUTTON.check_for_input(end_pos):  #gives new animal sound
                    draw_game()
                        
                elif PREDICT_BUTTON.check_for_input(end_pos): #when drawing is finished gives the image to AI model and it predicts what animal was drawn and gives score based on it's accuracy
                    gray_image = process_image()    #processing image to desired format (grayscale) and returns it
                    test = crop_image(gray_image)   #crops the image to desired format (square) and returns it
                    test = cv2.resize(test, (28,28), interpolation=cv2.INTER_AREA)  #resizes the image to 28x28 pixels so the model can use it
                    cv2.imwrite('image0.jpg', test)
                    _, test = cv2.threshold(test, 10, 255, cv2.THRESH_BINARY)
                    cv2.imwrite('image1.jpg', test)
                    kernel = np.ones((1, 1), np.uint8)  # Define a kernel for morphological operations
                    test = cv2.erode(test, kernel, iterations=10)   # Apply morphological operations to thin the edges
                    test = test/255.0
                    test = np.expand_dims(test, axis=0)
                    with open('encoder.pickle','rb') as f:
                        encode=pickle.load(f)
                    prediction = model.predict(test)
                    max_index = np.argmax(prediction)
                    one_hot_encoded = np.zeros_like(prediction)
                    one_hot_encoded[0][max_index] = 1
                    predicted_variables = encode.inverse_transform(np.reshape(one_hot_encoded,(1,-1)))[0][0]
                    max_value = round((prediction.max() * 100), 1)  #calculates the points for player
                    if predicted_variables == sound_to_str(sound_path): #if the prediction matches the sound
                        predict_text = get_font(25).render(f"You drew a {predicted_variables}", True, "Black", "White") 
                        predict_rect = predict_text.get_rect(center = (screen_width/2 -100, screen_height - 100))
                        screen.blit(predict_text, predict_rect)
                        points_text = get_font(25).render(f"Points: {max_value}/100", True, "Black", "White")
                        points_rect = points_text.get_rect(center = (screen_width/2 -100, screen_height - 50))
                        screen.blit(points_text, points_rect)
                    elif predicted_variables != sound_to_str(sound_path):   #if the prediction doesn't match the sound
                        fail_text = get_font(25).render(f"You were supposed to draw {sound_to_str(sound_path)}, Try again!", True, "Black", "White")
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
    pygame.time.wait(10)  # Add a small delay to avoid high CPU usage 
    
def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()
        
        screen.blit(BG, (0, 0)) #adds background image

        OPTIONS_TEXT = get_font(menu_font).render("Input:", True, "Black")  #defines the text
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(screen_width/2, screen_height/10))
        screen.blit(OPTIONS_TEXT, OPTIONS_RECT) #adds text to options screen
        #defines and adds all the buttons to options screen
        OPTIONS_MOUSE = Button(image=None, pos=(screen_width/2, screen_height/2 - screen_height/15 *3), 
                            text_input="MOUSE", font=get_font(menu_font), base_color="Black", hovering_color="White")
        OPTIONS_TOUCHSCREEN = Button(image=None, pos=(screen_width/2, screen_height/2 - screen_height/15), 
                            text_input="TOUCHSCREEN", font=get_font(menu_font), base_color="Black", hovering_color="White")
        OPTIONS_IRPEN = Button(image=None, pos=(screen_width/2, screen_height/2 + screen_height/15), 
                            text_input="IR PEN", font=get_font(menu_font), base_color="Black", hovering_color="White")
        OPTIONS_BACK = Button(image=None, pos=(screen_width/2, screen_height/2 + screen_height/15 *3), 
                            text_input="BACK", font=get_font(menu_font), base_color="Black", hovering_color="White")

        for button in [OPTIONS_MOUSE, OPTIONS_TOUCHSCREEN, OPTIONS_IRPEN, OPTIONS_BACK]:
            button.change_color(OPTIONS_MOUSE_POS)
            button.update(screen)


        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_MOUSE.check_for_input(OPTIONS_MOUSE_POS): #option for mouse mode of the game
                    draw_game()
                if OPTIONS_TOUCHSCREEN.check_for_input(OPTIONS_MOUSE_POS):  #option for built in touchscreen mode of the game
                    draw_game()
                if OPTIONS_IRPEN.check_for_input(OPTIONS_MOUSE_POS):    #option for IR pen mode of the game
                    draw_gameIR()
                    print("main done")
                if OPTIONS_BACK.check_for_input(OPTIONS_MOUSE_POS): #exits to main menu
                    main_menu()

        pygame.display.update()

def main_menu():
    music() #adds music to main menu (and options)
    while True:
        screen.blit(BG, (0,0))  #adds background image to main menu

        MENU_MOUSE_POS = pygame.mouse.get_pos()
        #defines and adds logo to menu screen
        logo_rect = logo.get_rect()
        logo_rect.center = (screen_width / 2, screen_height /6)
        screen.blit(logo, logo_rect.topleft)
        #defines and adds buttons to menu screen
        PLAY_BUTTON = Button(image=None, pos=(screen_width/2, screen_height/2 - screen_height/15), 
                            text_input="PLAY", font=get_font(menu_font), base_color="#000000", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(screen_width/2, screen_height/2 + screen_height/15), 
                            text_input="QUIT", font=get_font(menu_font), base_color="#000000", hovering_color="White")
        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.change_color(MENU_MOUSE_POS)
            button.update(screen)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.check_for_input(MENU_MOUSE_POS): #enters to input selection
                    options()
                if QUIT_BUTTON.check_for_input(MENU_MOUSE_POS): #exits the game
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

def music(): #music player
    pygame.mixer.music.load("assets/music.mp3")
    pygame.mixer.music.play(loops=-1)
    pygame.mixer.music.set_volume(0.3)
	
pygame.init()
clock = pygame.time.Clock()
#gets screen size directly ffrom operating system and adjusts the window to fullscreen mode
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height
screen = pygame.display.set_mode((screen_width, screen_height))
#scale fonts for different screen sizes
font_size = int(screen_height/40)           
menu_font = int(screen_height/10)           
pygame.display.set_caption('Sketchimals')   #adds the game name to the window
background_color = pygame.Color('White')    #white canvas for drawing
model = load_model("12_classes.h5") #loads the AI model
load_logo = pygame.image.load("assets/logo.png")    #loads the logo used in main menu
logo = pygame.transform.scale(load_logo, (screen_width/2, screen_width/2 * load_logo.get_height() / load_logo.get_width())) #scales the logo to look good in different screen sizes
BG = pygame.transform.scale(pygame.image.load("assets/background1.png"), (screen_width, screen_height)) #adjust the background image to fit all screensizes
pygame.mixer.init() #initializes the mixer for sounds

# rectangles for "AI"s guess and points
predict_rect = pygame.Rect(screen_width/2, screen_height - screen_height/10, screen_width/6, screen_height/20)
points_rect = pygame.Rect(screen_width/2, screen_height - screen_height/18, screen_width/6, screen_height/20)

while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    main_menu()      
    pygame.display.flip()


#Pig, elephant, cow, frog, monkey, dolphin, parrot, snake, dog, cat, sheep, horse
