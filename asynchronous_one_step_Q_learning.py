#!/usr/bin/env python
import multiprocessing as threading
import tensorflow as tf
import cv2
import sys, os
sys.path.append("Wrapped-Game-Code/")
import random
import numpy as np
import time
from launcher import *
from shared_utils import SharedCounter, Barrier
import json
import glob

parameters = process_args(sys.argv[1:], Defaults)

#Shared global parameters
TMAX = 5000000
T = SharedCounter(0)
It = 40000
Iasync = 5
WISHED_SCORE = 1000

GAMMA = parameters.discount # Decay rate of past observations
OBSERVE = 5. # Timesteps to observe before training
EXPLORE = 400000. # Frames over which to anneal epsilon
# FINAL_EPSILONS = [0.01, 0.01, 0.05] # Final values of epsilon
# INITIAL_EPSILONS = [0.4, 0.3, 0.3] # Starting values of epsilon
FINAL_EPSILONS = [0.1, 0.01, 0.5] # Final values of epsilon
INITIAL_EPSILONS = [1.0, 1.0, 1.0] # Starting values of epsilon
EPSILONS = len(INITIAL_EPSILONS)
EPSILONS_CHOICE_PROB = [0.4, 0.3, 0.3]
from bisect import bisect
def weighted_choice(rng, choices):
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = rng.rand() * total
    i = bisect(cum_weights, x)
    return values[i]

ale = load_ale(parameters, np.random.RandomState()) # loads number of actions
RESIZED_WIDTH, RESIZED_HEIGHT = Defaults.RESIZED_WIDTH, Defaults.RESIZED_HEIGHT

ACTIONS = len(ale.getMinimalActionSet())
GAME = ".".join(parameters.rom.split("/"))
CHECKPOINT_DIR = GAME+"-checkpoint/"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

T_file_name = CHECKPOINT_DIR + "/T.save"
def save_global_step():
    with open(T_file_name, "w+") as save_t_file:
        save_t_file.write(str(T.value()))

def restore_global_step():
    """ Only called once at start up, no locking needed """
    if os.path.exists(T_file_name):
        with open(T_file_name) as save_t_file:
            T.val.value =  int(save_t_file.read())
    print "Starting from Global T = {0}".format(T.value())
restore_global_step()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([3136, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, RESIZED_WIDTH, RESIZED_HEIGHT, 4], name="s")

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_conv3, [-1, 3136])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    return s, readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

def copyTargetNetwork(sess):
    print "Copied target network"
    sess.run(copy_Otarget)

class ActorLearner(object):
    CROP_OFFSET = 8
    def __init__(self, thread_id, target, parameters, lock, barrier, testing=False):
        self.lock = lock
        self.barrier = barrier
        self.thread_id = thread_id
        self.target = target
        self.rng = np.random.RandomState(123456 + thread_id * 13)
        self.ale = load_ale(parameters, self.rng, update_disp=thread_id==0 and parameters.update_disp)
        self.min_action_set = self.ale.getMinimalActionSet()
        self.buffer_length = 2
        self.buffer_count = 0
        self.resized_width = Defaults.RESIZED_WIDTH
        self.resized_height = Defaults.RESIZED_HEIGHT
        self.width, self.height = ale.getScreenDims()
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)
        self.frame_skip = parameters.frame_skip
        self.resize_method = parameters.resize_method
        self.max_start_nullops = parameters.max_start_nullops
        self.death_ends_episode = parameters.death_ends_episode
        self.thread_id = thread_id
        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.save_file_name =CHECKPOINT_DIR + "/t-{0}-checkpoint.save".format(thread_id)
        self.testing = testing

    def init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""
        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self.act(0) # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self.act(0)
        self.act(0)

    def act(self, action_idx):
        reward = self.ale.act(self.min_action_set[action_idx])
        index = self.buffer_count % self.buffer_length

        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])

        self.buffer_count += 1
        return reward

    def step(self, action_idx):
        """ action_idx is an integer encoding the ale action"""
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.act(action_idx)
        self.terminal_lol = (self.death_ends_episode and not self.testing and
                                 self.ale.lives() < self.start_lives)
        terminal = self.ale.game_over() or self.terminal_lol
        return reward, terminal

    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - ActorLearner.CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')

    def load_save_file(self, saved_name):
        with open(saved_name) as save_file:
            saved = json.load(save_file)
        epsilon = saved["epsilon"]
        INITIAL_EPSILON = saved["INITIAL_EPSILON"]
        FINAL_EPSILON = saved["FINAL_EPSILON"]
        t = saved["t"]
        return epsilon, INITIAL_EPSILON, FINAL_EPSILON, t

    def save_thread_progress(self, epsilon, INITIAL_EPSILON, FINAL_EPSILON, t):
        d = {
                "epsilon": epsilon,
                "INITIAL_EPSILON": INITIAL_EPSILON,
                "FINAL_EPSILON": FINAL_EPSILON,
                "t": t
            }
        with open(self.save_file_name, "w+") as save_file:
            json.dump(d, save_file)
            save_file.flush()

    def recover_or_init(self):
        if os.path.exists(self.save_file_name):
            return self.load_save_file(self.save_file_name)
        else:
            saves = glob.glob(CHECKPOINT_DIR + "/t-*-checkpoint.save")
            if len(saves) > 0:
                # we are recovering, but last run had less thread then current
                random_saved_name = self.rng.choice(saves)
                return self.load_save_file(random_saved_name)
            else:
                epsilon_index = weighted_choice(self.rng, 
                                                zip(range(EPSILONS), 
                                                EPSILONS_CHOICE_PROB))
                INITIAL_EPSILON = INITIAL_EPSILONS[epsilon_index]
                FINAL_EPSILON =  FINAL_EPSILONS[epsilon_index]
                epsilon = INITIAL_EPSILON
                return epsilon, INITIAL_EPSILON, FINAL_EPSILON, 0

    def learn(self):
        # We use global shared O parameter vector
        # We use global shared Otarget parameter vector
        # We use global shared counter T, and TMAX constant
        global TMAX, T

        # Initialize session and variables
        sess = tf.Session(self.target)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        if self.thread_id == 0:
            checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
            # Initialize target network weights
            copyTargetNetwork(sess)

        self.barrier.wait()

        # Initialize network gradients
        s_j_batch = []
        a_batch = []
        y_batch = []

        self.init_episode()
        self.start_lives = self.ale.lives()
        x_t = self.get_observation()
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
        aux_s = s_t

        epsilon, INITIAL_EPSILON, FINAL_EPSILON, t = self.recover_or_init()
        EPSILON_STEP = (INITIAL_EPSILON - FINAL_EPSILON) * parameters.np / EXPLORE

        print "THREAD ", self.thread_id, "STARTING...", "EXPLORATION POLICY => INITIAL_EPSILON:", INITIAL_EPSILON, ", FINAL_EPSILON:", FINAL_EPSILON

        # Initialize thread step counter
        score = 0
        while T.value() < TMAX and score < WISHED_SCORE:
            # Choose an action epsilon greedily
            readout_t = O_readout.eval(session = sess, feed_dict = {s : [s_t]})
            a_t = np.zeros([ACTIONS])
            if self.rng.rand() <= epsilon or t <= OBSERVE:
                action_index = self.rng.randint(0, ACTIONS)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # Scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= EPSILON_STEP

            # Run the selected action and observe next state and reward
            r_t, terminal = self.step(action_index)
            x_t1 = self.get_observation()
            x_t1 = x_t1.reshape(x_t1.shape + (1,))
            aux_s = np.delete(s_t, 0, axis = 2)
            s_t1 = np.append(aux_s, x_t1, axis = 2)

            # Accumulate gradients
            readout_j1 = Ot_readout.eval(session = sess, feed_dict = {st : [s_t1]})
            if terminal:
                y_batch.append(r_t)
            else:
                y_batch.append(r_t + GAMMA * np.max(readout_j1))

            a_batch.append(a_t)
            s_j_batch.append(s_t)

            # Update the old values
            s_t = s_t1
            t += 1
            score += r_t

            # Update the Otarget network
            T.lock.acquire()
            T.val.value += 1
            T_val = T.val.value
            T.lock.release()
            if T_val % It == 0:
                copyTargetNetwork(sess)

            # Update the O network
            if t % Iasync == 0 or terminal:
                if s_j_batch:
                    # Perform asynchronous update of O network
                    train_O.run(session = sess, feed_dict = {
                           y : y_batch,
                           a : a_batch,
                           s : s_j_batch})

                #Clear gradients
                s_j_batch = []
                a_batch = []
                y_batch = []

            # Save progress every 5000 iterations
            if t % 5000 == 0:
                self.save_thread_progress(epsilon, INITIAL_EPSILON, FINAL_EPSILON, t)
                if self.thread_id == 0:
                    saver.save(sess, CHECKPOINT_DIR + GAME + '-dqn', global_step = T.value())
                    save_global_step()

            if terminal:
                self.init_episode()
                self.start_lives = self.ale.lives()
                # Print info
                state = ""
                if t <= OBSERVE:
                    state = "observe"
                elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                    state = "explore"
                else:
                    state = "train"
                print "THREAD:", self.thread_id, "/ TIME", T.value(), "/ TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t), "/ SCORE", score
                score = 0    

def process_run_learner(thread_id, server_target, lock, barrier):
    learner = ActorLearner(thread_id, server_target, parameters, lock, barrier)
    learner.learn()

# We create the shared global networks
# O network
s, O_readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2 = createNetwork()

# Training node
a = tf.placeholder("float", [None, ACTIONS], name="a")
y = tf.placeholder("float", [None], name="y")
O_readout_action = tf.reduce_sum(tf.mul(O_readout, a), reduction_indices=1)
cost_O = tf.reduce_mean(tf.square(y - O_readout_action))
train_O = tf.train.RMSPropOptimizer(learning_rate=parameters.learning_rate, 
                                    decay=parameters.rms_decay, 
                                    epsilon=parameters.rms_epsilon,
                                    momentum=parameters.momentum).minimize(cost_O)
# Otarget network
st, Ot_readout, W_conv1t, b_conv1t, W_conv2t, b_conv2t, W_conv3t, b_conv3t, W_fc1t, b_fc1t, W_fc2t, b_fc2t = createNetwork()
copy_Otarget = [W_conv1t.assign(W_conv1), b_conv1t.assign(b_conv1), W_conv2t.assign(W_conv2), b_conv2t.assign(b_conv2), W_conv3t.assign(W_conv3), b_conv3t.assign(b_conv3), W_fc1t.assign(W_fc1), b_fc1t.assign(b_fc1), W_fc2t.assign(W_fc2), b_fc2t.assign(b_fc2)]

if __name__ == "__main__":
    # Start a local server
    server = tf.train.Server.create_local_server()
    # Start n concurrent actor threads
    lock = threading.Lock()
    barrier = Barrier(parameters.np)
    threads = list()
    for i in range(parameters.np):
        t = threading.Process(target=process_run_learner, args=(i, server.target, lock, barrier))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()

    print "ALL DONE!!"
