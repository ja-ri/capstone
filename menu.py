import pygame, sys
from time import sleep
from win32api import GetSystemMetrics

def get_font(size): 
    return pygame.font.Font("font.ttf", size)

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
    
    while True: 
        for event in pygame.event.get():
            (a, s) = pygame.mouse.get_pos() 
            if event.type == pygame.MOUSEMOTION and a >= 100:
                if event.buttons[0]:  
                    last = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    pygame.draw.line(screen, color, last, event.pos, size)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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

predict_rect = pygame.Rect(0, 300, 100, 50)
eraser_rect = pygame.Rect(0, 350, 100, 50)
black_rect = pygame.Rect(0, 400, 100, 50)
red_rect = pygame.Rect(0, 450, 100, 50)
green_rect = pygame.Rect(0, 500, 100, 50)
blue_rect = pygame.Rect(0, 550, 100, 50)
clear_rect = pygame.Rect(0, 600, 100, 50)

pygame.init()
clock = pygame.time.Clock()
screen_width = GetSystemMetrics(0)
screen_height = GetSystemMetrics(1)
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
BG = pygame.image.load("Background.png")
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

