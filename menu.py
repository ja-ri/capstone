from time import sleep
import pygame
import pygame_menu
from pygame_menu import themes
 
pygame.init()
surface = pygame.display.set_mode((600, 400))
 
def start_the_game():
    pass
 
mytheme = pygame_menu.themes.Theme(background_color=(0, 0, 0, 0), title_background_color = (4, 47, 126),
                title_font_shadow=True, widget_padding=25)
                
mainmenu = pygame_menu.Menu('Capstone', 600, 400, theme=mytheme)
mainmenu.add.button('Start', start_the_game)
mainmenu.add.button('Quit', pygame_menu.events.EXIT)
pygame_menu.widgets.HighlightSelection(border_width=1, margin_x=16, margin_y=8)
  
mainmenu.mainloop(surface)
