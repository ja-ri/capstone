import pygame
import sys


def start_btn():
    global game_state
    global btn_rect
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if btn_rect.collidepoint(event.pos):
            game_state = "draw_game"


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
    text_rect = btn_text.get_rect(
        center=(btn_surface.get_width()/2, btn_surface.get_height()/3))
    screen.blit(title, (screen_width/3, 80))
    screen.blit(instructions, (screen_width/3, 160))
    btn_surface.blit(btn_text, text_rect)
    screen.blit(btn_surface, (btn_rect.x, btn_rect.y))
    pygame.display.update()


def draw_game():
    global background_color
    screen.fill((background_color))
    drawing = False
    last_pos = None
    if event.type == pygame.MOUSEMOTION:
        if drawing:
            mouse_pos = pygame.mouse.get_pos()
            if last_pos is not None:
                pygame.draw.line(screen, last_pos, mouse_pos, 1)
            last_pos = mouse_pos
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pos = (0, 0)
            drawing = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
    pygame.display.update()


pygame.init()
clock = pygame.time.Clock()
screen_width = 1280
screen_height = 768
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Capstone Project')
background_color = pygame.Color('White')
btn_rect = pygame.Rect(600, 600, 100, 100)
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
