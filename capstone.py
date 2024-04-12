import pygame
from time import sleep
import sys
import pygame_menu
from pygame_menu import themes
          
            
def drawButtons():

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
    global background_color, btn_rect
    screen.fill((background_color))
    mainmenu = pygame_menu.Menu('Capstone', 600, 400, theme=mytheme)
    mainmenu.add.button('Start', draw_game)
    mainmenu.add.button('Quit', pygame_menu.events.EXIT)
    pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
  
    mainmenu.mainloop(screen)
    pygame.display.update()
    

def draw_game():
    global background_color, predict_rect, eraser_rect
    screen.fill((background_color))
    color = 'Black'
    size = 4
    drawButtons()
    
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
                    size = 4
               
                elif green_rect.collidepoint(event.pos):
                    color = 'Green'
                    size = 4
                    
                elif blue_rect.collidepoint(event.pos):
                    color = 'Blue'
                    size = 4
                
                elif black_rect.collidepoint(event.pos):
                    color = 'Black'
                    size = 4
                    
                elif eraser_rect.collidepoint(event.pos):
                    color = 'White'
                    size = 25

                elif clear_rect.collidepoint(event.pos):
                    screen.fill('White')
                    drawButtons()

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
