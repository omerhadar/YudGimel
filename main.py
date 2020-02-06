import random
import os
import pygame
import pygame.gfxdraw
from DQA import *
import numpy as np
from keras.utils import to_categorical
from math import *
from pygame.locals import *
import time
import sys

SPEED = 60      # frames per second setting
WINWIDTH = 1280  # width of the program's window, in pixels
WINHEIGHT = 720  # height in pixels
RADIUS = 5      # radius of the circles
PLAYERS = 1      # number of players
TURN_RATE = 5  # change of angle per turn move

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9975  # decay percentage per run
MIN_EPSILON = 0.001  # minimum value of epsilon

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# check run mode
if len(sys.argv) < 2:
    print("need more system arguments")
    exit(2)
AI = int(sys.argv[1])
# initialize agent
agent = DQNAgent()


def main():
    global epsilon
    os.environ['SDL_VIDEO_CENTERED'] = '0'
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
    # main loop
    global FPS_CLOCK, SCREEN, DISPLAYSURF
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
    DISPLAYSURF = pygame.Surface(SCREEN.get_size())
    pygame.display.set_caption('Nope!')

    if AI == 2:
        if len(sys.argv) < 3:
            print("give weights file name")
            exit(3)
        agent.load_model(sys.argv[2])

    ep_rewards = []
    games_counter = 0
    while True:
        ep_rewards.append(run_game(games_counter))
        games_counter += 1
        if not games_counter % AGGREGATE_STATS_EVERY or games_counter == 0:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(sys.argv[0][:-sys.argv[0][::-1].index('/')] +
                                 f'models/{MODEL_NAME}__{max_reward:_>7.2f}max' +
                                 f'_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
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
        self.x = random.randrange(50, WINWIDTH - 165)
        self.y = random.randrange(50, WINHEIGHT - 50)
        self.angle = random.randrange(0, 360)
        #self.x = 500
        #self.y = 500
        #self.angle = 0

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
        for i in range(self.angle - 90, self.angle + 90, 10):
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


def run_game(count):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    DISPLAYSURF.fill(BLACK)

    # Renew border
    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, 2), (WINWIDTH - 2, WINHEIGHT - 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (2, 2), (2, WINHEIGHT - 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, 2), (2, 2), 10)
    pygame.draw.line(DISPLAYSURF, RED, (WINWIDTH - 2, WINHEIGHT - 2), (2, WINHEIGHT - 2), 10)

    # initialize local variables
    moves = 0
    run = True
    reward = 0

    # generating players
    player1 = Player()
    player1.gen()

    # getting old state
    state_old = agent.get_state(player1, DISPLAYSURF)

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

        player1.colour = RED
        player1.draw(False)

        # steering
        if not AI:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player1.angle -= TURN_RATE
            if keys[pygame.K_RIGHT]:
                player1.angle += TURN_RATE
        else:
            if np.random.random() < epsilon and AI == 1:
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

        # check if player is dead after moving
        if player1.is_dead():
            reward -= 100
            print("game", count, "ended with rewards", reward, "epsilon", epsilon)
            run = False

        if AI:
            # get new state
            state_new = agent.get_state(player1, DISPLAYSURF)

            # increase reward
            reward += 1

            # train and update memory
            agent.update_replay_memory((state_old, final_move, reward, state_new, not run))
            agent.train(not run, step)

            # set the old state as the new one
            state_old = state_new

        # draw the yellow head
        player1.colour = YELLOW
        player1.draw()

        # option to quit if human
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
        if not AI:
            FPS_CLOCK.tick(SPEED)
    return reward


if __name__ == '__main__':
    main()
