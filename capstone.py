import pygame
import sys


def start_btn():
    global game_state
    global btn_rect
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if btn_rect.collidepoint(event.pos):
            game_state = "draw_game"
            
            
def red_btn():
    global red_rect, pen_color
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if red_rect.collidepoint(event.pos):
            pen_color = 2

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
    
    
    
    while True: 
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                isPressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                isPressed = False
            elif event.type == pygame.MOUSEMOTION and isPressed == True:         
                ( x, y ) = pygame.mouse.get_pos()
                if pen_color == 1:
                    drawCircleB( screen, x, y )
                elif pen_color == 2:
                    drawCircleR(screen, x, y)
                elif pen_color == 3:
                    drawCircleG(screen, x, y)
                elif pen_color == 4:
                    drawCircleBl(screen, x, y)
                elif pen_color == 0:
                    drawCircleW(screen, x, y)
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
pen_color = 1       #colors: 0=eraser 1=black 2=red 3=green 4=blue
btn_rect = pygame.Rect(600, 600, 100, 100)
predict_rect = pygame.Rect(0, 300, 100, 100)
eraser_rect = pygame.Rect(0, 350, 100, 100)
black_rect = pygame.Rect(0, 400, 100, 100)
red_rect = pygame.draw.rect(screen, 'Yellow', pygame.Rect(0, 450, 100, 100))
green_rect = pygame.Rect(0, 500, 100, 100)
blue_rect = pygame.Rect(0, 550, 100, 100)
clear_rect = pygame.Rect(0, 600, 100, 100)
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
        red_btn()
        

    pygame.display.flip()
