# -*- coding: utf-8 -*-

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import save
from numpy import load
import matplotlib
matplotlib.use("TkAgg")
np.set_printoptions(suppress=True)

class cls_Environment:

    def __init__(self, m=1, l=1, mu=0.01, m_s=2 * pi, m_t=5, dt=0.001, g=9.8):
        self.max_speed = m_s  # + or - this value
        self.max_torque = m_t  # + or - this value
        self.m = m
        self.l = l
        self.g = g
        self.mu = mu
        self.dt = dt

    def reset(self):
        th = pi
        thdot = 0
        th = th + ((th + np.pi) // (2 * np.pi)) * (-2 * np.pi)
        self.state = np.array([th, thdot])
        # th := theta [th, thdot] pendulum hanging down is pi,
        # but following our rescaling mechanism "any angle -> [-pi, pi[", it's rescaled to -pi
        return self.state

    def step(self, a, dt_steps=100):
        cost = self._calc_cost(self.state)
        a = np.clip(a, -self.max_torque, self.max_torque)
        for i in range(dt_steps):
            self._solve_ode(a)
        s_next = self.state
        cost_next = self._calc_cost(s_next)
        return s_next, cost, cost_next

    def _solve_ode(self, a):
        th, thdot = self.state  # th := theta
        thdotdot = (1 / self.m * self.l ** 2) * (-self.mu * thdot + self.m * self.g * self.l * np.sin(th) + a)
        thdot = thdot + thdotdot * self.dt
        thdot = np.clip(thdot, -self.max_speed, self.max_speed)
        th = th + thdot * self.dt + (1 / 2) * thdotdot * self.dt ** 2
        # th = self._rescale_th(th) <- I guess we don't need this anymore?
        th = th + ((th + np.pi) // (2 * np.pi)) * (-2 * np.pi)
        self.state = [th, thdot]

    def _calc_cost(self, s_next):
        cost = -np.abs(s_next[0])
        return cost


class Partition:

    def __init__(self, empty=False, threshold_var=0.6, threshold_n=3):
        # 0:number partition,
        # 1:thmin(incl.), 2:thmax(excl.),
        # 3:thdotmin(incl.), 4:thdotmax(excl.),
        # 5: amin(incl.), 6:amax(excl.),
        # 7:n,
        # 8:Qmed,
        # 9:Variance
        if empty:
            self.partition = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
        else:
            # First partition is 0-0.5 and 0.5-1 all values. That means
            # th: -pi to 0 and 0 to pi,thdot: -2*pi to 0 and 0 to 2*pi,a: -5 to 0 and 0 to 5
            # initial max-limits set to >1 (1.01), so that we are always strictly smaller than the upper bound
            self.partition = np.array([[0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 10],
                                       [1, 0.5, 1.01, 0, 0.5, 0, 0.5, 0, 0, 10],
                                       [2, 0, 0.5, 0.5, 1.01, 0, 0.5, 0, 0, 10],
                                       [3, 0.5, 1.01, 0.5, 1.01, 0, 0.5, 0, 0, 10],
                                       [4, 0, 0.5, 0, 0.5, 0.5, 1.01, 0, 0, 10],
                                       [5, 0.5, 1.01, 0, 0.5, 0.5, 1.01, 0, 0, 10],
                                       [6, 0, 0.5, 0.5, 1.01, 0.5, 1.01, 0, 0, 10],
                                       [7, 0.5, 1.01, 0.5, 1.01, 0.5, 1.01, 0, 0, 10]])

        self.threshold_var = threshold_var
        self.threshold_n = threshold_n

    def find_part(self, s, a):
        norm_s, norm_a = self._normalize_data_(s, a)

        # Find the partition where the s,a lay
        for i in range(len(self.partition)):
            if ((self.partition[i][1] <= norm_s[0] < self.partition[i][2]) and
                    (self.partition[i][3] <= norm_s[1] < self.partition[i][4]) and
                    (self.partition[i][5] <= norm_a < self.partition[i][6])):
                return i

    #        """ #option without loop (not tested):
    #        i = np.where((self.partition[:, 1] < norm_s[0]) * (self.partition[:, 2] >= norm_s[0]) *
    #                       (self.partition[:, 3] < norm_s[1]) * (self.partition[:, 4] >= norm_s[1]) *
    #                       (self.partition[:, 5] < norm_a) * (self.partition[:, 6] >= norm_a))[0][0]
    #                       """

    def _normalize_data_(self, s, a):
        # Transforms the input into 0-1 scalar
        s[0] = np.interp(s[0], (-np.pi, np.pi), (0, 1))
        s[1] = np.interp(s[1], (-np.pi * 2, np.pi * 2), (0, 1))
        a = np.interp(a, (-5, 5), (0, 1))
        return s, a

    def split(self, i):
        # Calculate the size of the partition to get the largest difference
        dif_th = self.partition[i][2] - self.partition[i][1]
        dif_thdot = self.partition[i][4] - self.partition[i][3]
        dif_a = self.partition[i][6] - self.partition[i][5]

        # Split by theta
        if dif_th >= dif_thdot and dif_th >= dif_a:
            # Update the actual partition as the lower half of it
            self.partition[i][2] -= dif_th / 2
            self.partition[i][7] = 0
            # Create the new partition as the bigger half of it and append it
            new_row = np.append([len(self.partition), self.partition[i][2], self.partition[i][2] + dif_th / 2],
                                [self.partition[i][3:]])
            self.partition = np.append(self.partition, [new_row], axis=0)

        # Split by theta dot
        elif dif_thdot > dif_th and dif_thdot > dif_a:
            self.partition[i][4] -= dif_thdot / 2
            self.partition[i][7] = 0
            new_row = np.append(len(self.partition), [self.partition[i][1:3]])
            new_row = np.append(new_row, [self.partition[i][4], self.partition[i][4] + dif_thdot / 2])
            new_row = np.append(new_row, [self.partition[i][5:]])
            self.partition = np.append(self.partition, [new_row], axis=0)

        # Split by action
        else:
            self.partition[i][6] -= dif_a / 2
            self.partition[i][7] = 0
            new_row = np.append(len(self.partition), [self.partition[i][1:5]])
            new_row = np.append(new_row, [self.partition[i][6], self.partition[i][6] + dif_a / 2])
            new_row = np.append(new_row, [self.partition[i][7:]])
            self.partition = np.append(self.partition, [new_row], axis=0)

    def var_res(self, q, s, a):
        # Find partition
        i = self.find_part(s, a)
        # Update parameters (n,Q,S) of the partition i
        self.partition[i][7] += 1
        #        print('lr', learning_rate(self.partition[i][7]))
        self.partition[i][8] += learning_rate(self.partition[i][7]) * (q - self.partition[i][8])
        self.partition[i][9] += learning_rate(self.partition[i][7]) * (
                    np.square(q - self.partition[i][8]) - self.partition[i][9])

        # print('threshold:', np.square(q-self.partition[i][8]))
        if ((np.square(q - self.partition[i][8]) > self.threshold_var) and
                (self.partition[i][7] > self.threshold_n)):
            self.split(i)


def select_action(partition, s, n_draws=21):
    # randomly draw q for all discretised actions in env.state
    actions = np.linspace(-5, 5, num=n_draws)
    q_rand = np.array([actions, np.zeros(n_draws)]).transpose()
    # print('draw ', n_draws, ' actions:\n')
    for item in q_rand:
        i = partition.find_part(np.copy(s), item[0])
        # print(partition.partition[i])
        item[1] = np.random.normal(
            partition.partition[i, 8],
            np.sqrt(partition.partition[i, 9]))
        # print(item)
    a = q_rand[np.argmax(q_rand[:, 1]), 0]

    return a


def calculate_value(partition, s, cost, gamma):
    # calculate the value q of current state. Only applies cost (input) and value of new state s' (which is the
    # current env.state, since env.state was updated incrementally to get to s').
    # The value of new state s' is equal to the maximum value of state s' across all possible actions a'.

    # normalize env.state
    norm_s, _a = partition._normalize_data_(np.copy(s), 0)

    partition_new_state = np.where((partition.partition[:, 1] <= norm_s[0]) *
                                   (partition.partition[:, 2] > norm_s[0]) *
                                   (partition.partition[:, 3] <= norm_s[1]) *
                                   (partition.partition[:, 4] > norm_s[1]))
    # print('partition of new state: \n', partition_new_state)
    Q_max = np.max(partition.partition[partition_new_state, 8])
    # print('Q_max: ', Q_max)
    q = cost + gamma * Q_max

    return q


def apply_policy(partition, s):
    # determine the best policy for given env.state,
    # since each Q-value is given for a range of actions, the mean action of the partition is taken

    s, _a = partition._normalize_data_(np.copy(s), 0)

    partition_new_state = partition.partition[np.where((partition.partition[:, 1] <= s[0]) *
                                                       (partition.partition[:, 2] > s[0]) *
                                                       (partition.partition[:, 3] <= s[1]) *
                                                       (partition.partition[:, 4] > s[1]))]
    # print('part new state', partition_new_state)
    # print('s', s)
    # print('part.part', partition.partition)
    norm_a = np.mean(partition_new_state[np.argmax(partition_new_state[:, 8])][5:7])
    a = np.interp(norm_a, (0, 1), (-5, 5))

    # cost = -np.abs(s[0]) not needed

    return a  # , cost


def learning_rate(t):
    a = 0.008
    b = 2
    return 1 / (a * t + b)


# test env without RL logic
def train_agent(partition, env, gamma=0.9, n_iterations=500):
    # To Do: replace this to lines of code with Q learning with variable resolution
    # action = np.random.choice(action_interval)/10

    env.reset()

    for i in range(n_iterations):
        # print('\n\n\n##########################################\n i:', i)
        s = np.copy(env.state)
        # print('state:', s)
        a = select_action(partition, s, 21)
        s_next, cost, _cost_next = env.step(a)
        # print('transition: ', s, '->', a, '->',s_next)
        q = calculate_value(partition, s_next, cost, gamma)
        # print('q:', q)
        partition.var_res(q, s, a)
        # print('new partition:\n', partition.partition)
        # print('check env.state:', env.state)

    return partition


def test_agent(partition, env, n_iterations=500):
    env.reset()
    accu_cost = 0
    # print('final partition:\n', partition.partition)

    for i in range(n_iterations):
        s = np.copy(env.state)
        # print('state:', s)
        a = apply_policy(partition, s)
        _s_next, cost, _cost_next = env.step(a)
        # print('\ntransition: ', s, '->', a, '->',_s_next, '\n')
        accu_cost += cost
        # print('check env.state:', env.state)
        # print('$$$ (iteration and accumulated):', cost, accu_cost)

        # I guess here goes the pendelum plot code?

    return accu_cost


def ani_agent(i):
    s = np.copy(env.state)
    a = apply_policy(partition, s)
    s_next, cost, _ = env.step(a)
    x, y = np.sin(s_next[0]), np.cos(s_next[0])
    line, = ax.plot((0, x), (0, y), "r")
    circle, = ax.plot(x, y, "ro")
    return line, circle


## MAIN

# Initialize
threshold_var = 0.6
threshold_n = 3
gamma = 0.9
env = cls_Environment(m=1, l=1, mu=0.01, m_s=2 * pi, m_t=5, dt=0.001)
partition = Partition(False, threshold_var, threshold_n)

# Initialize rewards array 100
rewards = np.zeros((100,))

# for-loop 100:
for i in range(100):
    print(i)
    train_agent(partition, env, gamma)
    rewards[i] = test_agent(partition, env)
    print(rewards[i])

parti = partition.partition

# save('final_partition.npy', parti)
print(parti)
# Plot rewards array
r_plot = plt.figure(1)
plt.plot(rewards)

r_plot.show()

# Initialize plot for animation
fig = plt.figure(2)
ax = fig.add_subplot(1, 1, 1)
plt.axis([-2, 2, -2, 2])
ani_iterations = 100  # 10 sec

# Visualize the behaviour of the pendulum during 10 seconds
env.reset()
ani = FuncAnimation(fig, func=ani_agent, fargs=(), frames=ani_iterations, interval=100, blit=True, repeat=False)

plt.show()