from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
import threading
from math import *


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.01
        self.model = self.network()
        #self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, player, board):

        def thread_func(p, mod, b):
            if mod == 1:
                while b.get_at((p.x + int(5 * cos(radians(p.angle))),
                                p.y + int(5 * sin(radians(p.angle))))) != (255, 0, 0):
                    lst[mod] += 1
                    p.move()
            else:
                while not p.is_dead() and lst[mod] < 70:
                    lst[mod] += 1
                    if mod == 0:
                        p.angle -= 5
                    elif mod == 2:
                        p.angle += 5
                    p.move()

        lst = [0, 0, 0]

        ths = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(player.copy(), i, board))
            ths.append(t)
            t.start()

        for t in ths:
            t.join()

        return np.asarray(lst)

    def set_reward(self):
        self.reward += 1
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(units=10, activation='relu', input_dim=3))
        model.add(Dropout(0.15))
        model.add(Dense(units=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 3)))[0])
        target_f = self.model.predict(state.reshape((1, 3)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 3)), target_f, epochs=1, verbose=0)
