from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import TensorBoard
import tensorflow as tf
import random
import numpy as np
import threading
import time
from collections import deque

REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = 'model'
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200


class DQNAgent(object):

    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{time.time()}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def get_state(self, player, board):

        def thread_func(p, mod):
            while not p.is_dead() and lst[mod] < 45:
                lst[mod] += 1
                if mod == 0:
                    p.angle -= 5
                elif mod == 2:
                    p.angle += 5
                p.move()
            lst[mod] = 1 - (lst[mod] / 45)

        lst = [0, 0, 0]

        ths = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(player.copy(), i))
            ths.append(t)
            t.start()

        for t in ths:
            t.join()

        return np.asarray(lst)

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=50, activation='relu', input_dim=3))
        model.add(Dropout(0.15))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=3, activation='softmax'))
        opt = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=opt)
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape((1, 3)))[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[np.array(action).argmax()] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def load_model(self, file_name):
        self.model.load_weights(file_name)
        self.target_model.load_weights(file_name)


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)