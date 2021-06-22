import numpy as np
import tensorflow as tf
import datetime
import os
import matplotlib.pyplot as plt
from env import clsEnvironment
from clsNetwork import clsDQN

# comment out to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# function to train the Network
def train(cart_pole, QA1, QA2, actions, episodes, eps, copy_Q1_Q2):
    # reset env and get first state
    state = cart_pole.reset()

    # reset vars
    rewards = 0
    done = False

    L = np.empty(episodes)

    # loop to run trhough training episodes number of times
    for i in range(episodes):
        # get action
        action = QA1.get_action(state, eps)

        # housekeeping
        prev_state = state
        # get the next state with the action
        _, state, reward, quit = cart_pole.action_step(actions[action])
        # save reward
        rewards = rewards + reward

        # check if done
        if i == (episodes-1) or quit:
            done = True

        # create and store experience
        QA1.store_experience({'s': prev_state, 'a': action, 'r': reward, 's_1': state, 'done': done})
        # run training and get loss
        loss = QA1.train(QA2)
        #store loss
        L[i] = loss

        # update Q1 every copy_Q1_Q2 times
        if i % copy_Q1_Q2 == 0:
            QA2.copy_Q1_model(QA1)

        # if env quits, quit loop
        if quit:
            break
            
    # calc mean loss
    mean_loss = np.mean(L)
    return rewards, mean_loss, i


# function to test the NN
def test(QA1, cart_pole, actions, video_time, noise=0.0):
    # cals test steps
    N = int(video_time/cart_pole.dt)
    # reset env and get first state
    state = cart_pole.reset()

    # calc rewards
    rewards = 0
    # create array to store tranjectory
    States_Array = np.zeros((N, 4))

    # run test loop
    for i in range(N):
        t = round(i * cart_pole.dt)
        # store state
        States_Array[i, :] = state
        # get action
        action = QA1.get_action(state, 0)  # eps = 0, e.g. no randomness
        # run action in env with or without noise
        if int(t) == 5 and noise != 0.0:
            t, state, reward, abort = cart_pole.action_step(actions[action], noise)
        else:
            t, state, reward, abort = cart_pole.action_step(actions[action], 0)

        # save reward
        rewards = rewards + reward

    # print debug vals
    #print(f"Last state: {cart_pole.state}, Test: {i}, rewards {rewards}, done: {cart_pole.done}")
    return rewards, States_Array

# main functtion
def main(pack_param, epochs = 500, episodes = 100):
    # unpack params
    pole_m, cart_m, pole_l, x_min, x_max, x_dot_min, x_dot_max, th_min, th_max, th_dot_min, th_dot_max, \
    f_min, f_max, sim_inter, action_inter = pack_param

    # create cart_pole environment
    cart_pole = clsEnvironment(pole_m=pole_m, cart_m=cart_m, pole_l=pole_l, x_min=x_min, x_max=x_max,
                               x_dot_min=x_dot_min, x_dot_max=x_dot_max, th_min=th_min, th_max=th_max,
                               th_dot_min=th_dot_min, th_dot_max=th_dot_max, f_min=f_min, f_max=f_max,
                               sim_inter=sim_inter, action_inter=action_inter)
    #get number of state to create NN
    num_states = len(cart_pole.reset())

    # def actions
    actions = np.arange(f_min, f_max, 0.5)
    # create number of
    num_actions = len(actions)

    # create arrays to store log
    epoch_iters = np.empty(epochs)
    total_R = np.empty(epochs)

    # discount factor
    gamma = 0.9
    # defince lr functoin
    lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-4, power=0.5)
    # define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    # define NN structure
    hidden_layers = np.array([128, 256, 256, 256, 256])

    # def batch size
    batch_size = 512
    #def exp buffer size
    experience_buffer_size = 5000

    # init Q1 and Q2
    QA1 = clsDQN(num_states, num_actions, hidden_layers, gamma, optimizer, experience_buffer_size, batch_size)
    QA2 = clsDQN(num_states, num_actions, hidden_layers, gamma, optimizer, experience_buffer_size, batch_size)
    # copy so that wights of bothe networks are the same at init point
    QA2.copy_Q1_model(QA1)

    # how long shall the video tun?
    video_time = 10

    # run the training for n epcohs
    for n in range(epochs):
        # reset env before each training epoch
        cart_pole.reset()
        # run trating for episode times
        total_r, mean_loss, iters = train(cart_pole, QA1, QA2,actions, episodes, eps=0.25, copy_Q1_Q2=20)
        # save nr of iters
        epoch_iters[n] = iters
        # test how well policy works every 20 eipsodes
        if n % 20 == 0:
            # test
            total_r, States_Array = test(QA1, cart_pole, actions, video_time)
            # save reward
            total_R[n] = total_r
            # print progress
            print(f"Epoch: {n}, reward: {total_r}, loss: {mean_loss}, iters: {iters}")

    # call plot fn
    plot_jpg(epochs, total_R)

    # introduce noise / pertubation
    noise = 0
    if noise != 0:
        total_r, States_Array = test(QA1, cart_pole, actions, video_time, noise)

    # render the trajectory and make video
    cart_pole.render(States_Array, name=str("video") + str(video_time) + '_cart_pole_' + str(noise))

# plot the reward
def plot_jpg(epochs, total_R):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    plt.figure()
    plt.plot(np.arange(epochs), total_R, linewidth=1)
    plt.xlabel("Training_time")
    plt.ylabel("Accumulated R")
    plt.tight_layout()
    plt.savefig("plots/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_R.jpg", format="jpg")
    plt.close()

# main call fn
if __name__ == '__main__':

    #paramters from task:
    #pole mass
    pole_m = 0.5
    # cart mass
    cart_m = 0.5
    # pole lenght
    pole_l = 0.6

    # x limit
    x_min = -6.0
    # x limit
    x_max = 6.0
    # x dot lim
    x_dot_min = -10.0
    # x dot lim
    x_dot_max = 10.0
    # theta lim
    th_min = -np.pi
    # theta lim
    th_max = np.pi
    # theta dot lim
    th_dot_min = -10.0
    # theta dot lim
    th_dot_max = 10.0
    # f lim
    f_min = -10.0
    # f lim
    f_max = 10.0

    # simulation intervall. Nr of sim steps between 2 actions
    sim_inter = 0.01
    # in [s] when action can be performed
    action_inter = 0.1

    # pack params
    pack_param = pole_m, cart_m, pole_l, x_min, x_max, x_dot_min, x_dot_max, th_min, th_max, th_dot_min, th_dot_max, \
               f_min, f_max, sim_inter, action_inter

    # call main fn with how many epochs and episodes per epoch it should run.
    main(pack_param, epochs = 500, episodes = 100)
