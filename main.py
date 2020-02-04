import random
import os
import pygame
import pygame.gfxdraw
from DQA import DQNAgent
import numpy as np
from keras.utils import to_categorical
from math import *
from pygame.locals import *

SPEED = 60      # frames per second setting
WINWIDTH = 1280  # width of the program's window, in pixels
WINHEIGHT = 720  # height in pixels
RADIUS = 5      # radius of the circles
PLAYERS = 1      # number of players
TURN_RATE = 5

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
P1COLOUR = RED
P2COLOUR = GREEN
P3COLOUR = BLUE
YELLOW = (255, 255, 0)

AI = True
agent = DQNAgent()


def main():
    os.environ['SDL_VIDEO_CENTERED'] = '0'
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
    # main loop
    global FPS_CLOCK, SCREEN, DISPLAYSURF, MY_FONT
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
    DISPLAYSURF = pygame.Surface(SCREEN.get_size())
    pygame.display.set_caption('Nope!')

    games_counter = 0
    while True:
        rungame(games_counter)
        games_counter += 1
        SCREEN = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
        DISPLAYSURF = pygame.Surface(SCREEN.get_size())


class Player(object):
    # Class which can be used to generate random position and angle, to compute movement values and to draw player
    def __init__(self, x=0, y=0, angle=0, player=None):
        if player:
            self.running = True
            self.colour = None
            self.score = 0
            self.x = player.x
            self.y = player.y
            self.angle = player.angle
        else:
            self.running = True
            self.colour = None
            self.score = 0
            self.x = x
            self.y = y
            self.angle = angle

    def gen(self):
        # generates random position and direction
        #self.x = random.randrange(50, WINWIDTH - 165)
        #self.y = random.randrange(50, WINHEIGHT - 50)
        #self.angle = random.randrange(0, 360)
        self.x = 500
        self.y = 500
        self.angle = 0

    def copy(self):
        return Player(player=self)

    def move(self):
        # computes current movement
        self.x += int(RADIUS * cos(radians(self.angle)) / 1)
        self.y += int(RADIUS * sin(radians(self.angle)) / 1)

    def is_dead(self):
        if self.x < 5 or self.x > WINWIDTH - 5 or self.y < 5 or self.y > WINHEIGHT - 5:
            return True
        if self.colour == BLACK:
            return False
        for i in range(self.angle - 90, self.angle + 90, 5):
            if DISPLAYSURF.get_at((self.x + int(RADIUS * cos(radians(i))),
                                   self.y + int(RADIUS * sin(radians(i))))) == RED:
                return True
        return False

    def draw(self, aa=True):
        # drawing players
        if aa:
            pygame.gfxdraw.aacircle(DISPLAYSURF, self.x, self.y, RADIUS, self.colour)
        else:
            pygame.gfxdraw.circle(DISPLAYSURF, self.x, self.y, RADIUS, self.colour)
        pygame.gfxdraw.filled_circle(DISPLAYSURF, self.x, self.y, RADIUS, self.colour)


def rungame(count):
    global WINNER

    epsilon = 150 - count

    DISPLAYSURF.fill(BLACK)

    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, 2), (WINWIDTH - 2, WINHEIGHT - 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (2, 2), (2, WINHEIGHT - 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, 2), (2, 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, WINHEIGHT - 2), (2, WINHEIGHT - 2), 10)

    moves = 0
    run = True
    reward = 0

    # generating players
    player1 = Player()
    player1.gen()
    while run:
        moves += 1
        """
        # generating random holes
        hole = random.randrange(1, 50)
        if hole == 1 or flag:
            flag += 1
            if flag > 8:
                flag = -68
                player1.colour = RED
            elif flag < 0:
                flag += 1
                player1.colour = RED
            else:
                player1.colour = BLACK
        else:
            player1.colour = RED
        """
        state_old = agent.get_state(player1, DISPLAYSURF)

        player1.colour = RED
        player1.draw(False)

        if not AI:
            # steering
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player1.angle -= TURN_RATE
            if keys[pygame.K_RIGHT]:
                player1.angle += TURN_RATE
        else:
            if random.randint(0, 70) < epsilon:
                final_move = [0, 0, 0]
                final_move[random.randint(0, 2)] = 1
            else:
                pred = agent.model.predict(state_old.reshape((1, 3)))
                final_move = to_categorical(np.argmax(pred[0]), num_classes=3)

            if final_move[0] == 1:
                player1.angle -= TURN_RATE
            elif final_move[1] == 1:
                player1.angle += TURN_RATE

        player1.move()

        if player1.is_dead():
            reward -= 200
            print("game", count, "ended with rewards", reward)
            run = False

        if AI:
            state_new = agent.get_state(player1, DISPLAYSURF)

            reward += 1
            # if moves > 100 and moves % 70 == 0:
            #    reward += 10

            agent.train_short_memory(state_old, final_move, reward, state_new, player1.is_dead())

            agent.remember(state_old, final_move, reward, state_new, player1.is_dead())

        player1.colour = YELLOW
        player1.draw()

        if not AI:
            for event in pygame.event.get():
                if event.type == QUIT:
                    run = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        run = False

        # drawing all on the screen
        SCREEN.blit(DISPLAYSURF, (0, 0))
        pygame.display.update()
        # FPS_CLOCK.tick(SPEED)
    agent.replay_new(agent.memory)


if __name__ == '__main__':
    main()
