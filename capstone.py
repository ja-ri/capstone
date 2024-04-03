import pygame
import sys


def start_btn():
    global game_state
    global btn_rect
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if btn_rect.collidepoint(event.pos):
            game_state = "draw_game"
            
def drawCircleB( screen, x, y ):
    pygame.draw.circle( screen, 'Black', ( x, y ), 5 )
    
def drawCircleW( screen, x, y ):
    pygame.draw.circle( screen, 'White', ( x, y ), 10 )
    
def drawCircleBl( screen, x, y ):
    pygame.draw.circle( screen, 'Blue', ( x, y ), 5 )

def drawCircleR( screen, x, y ):
    pygame.draw.circle( screen, 'Red', ( x, y ), 5 )

def drawCircleG( screen, x, y ):
    pygame.draw.circle( screen, 'Green', ( x, y ), 5 )

def draw_start_menu():
    global background_color, btn_rect
    screen.fill((background_color))
    font1 = pygame.font.Font("freesansbold.ttf", 50)
    font2 = pygame.font.Font("freesansbold.ttf", 24)
    font3 = pygame.font.Font("freesansbold.ttf", 20)
    title = font1.render('Title', False, 'Black')
    instructions = font2.render('Instructions', False, 'Black')
    btn_surface = pygame.Surface((150, 50))
    btn_text = font3.render('Start', False, 'White')
    text_rect = btn_text.get_rect(center=(btn_surface.get_width()/2, btn_surface.get_height()/2))
    screen.blit(title, (screen_width/3, 80))
    screen.blit(instructions, (screen_width/3, 160))
    btn_surface.blit(btn_text, text_rect)
    screen.blit(btn_surface, (btn_rect.x, btn_rect.y))
    pygame.display.update()
    

def draw_game():
    global background_color, predict_rect, eraser_rect
    screen.fill((background_color))
    font4 = pygame.font.Font("freesansbold.ttf", 16)
    predict_surface = pygame.Surface((100, 50))
    predict_text = font4.render('Predict', False, 'White')
    text1_rect = predict_text.get_rect(center=(predict_surface.get_width()/2, predict_surface.get_height()/2))
    predict_surface.blit(predict_text, text1_rect)
    screen.blit(predict_surface, (predict_rect.x, predict_rect.y))
    eraser_surface = pygame.Surface((100, 50))
    eraser_text = font4.render('Eraser', False, 'White')
    text2_rect = eraser_text.get_rect(center=(eraser_surface.get_width()/2, eraser_surface.get_height()/2))
    eraser_surface.blit(eraser_text, text2_rect)
    screen.blit(eraser_surface, (eraser_rect.x, eraser_rect.y))
    
    
    while True: 
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                isPressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                isPressed = False
            elif event.type == pygame.MOUSEMOTION and isPressed == True:         
                ( x, y ) = pygame.mouse.get_pos()
                drawCircleB( screen, x, y )
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()


pygame.init()
clock = pygame.time.Clock()
screen_width = 1280
screen_height = 768
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
btn_rect = pygame.Rect(600, 600, 100, 100)
predict_rect = pygame.Rect(0, 300, 100, 100)
eraser_rect = pygame.Rect(0, 350, 100, 100)
game_state = "start_menu"


while True:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if game_state == "start_menu":
        draw_start_menu()
        start_btn()
    if game_state == "draw_game":
        draw_game()
        

    pygame.display.flip()
