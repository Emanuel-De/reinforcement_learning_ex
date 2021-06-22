import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# env class
class clsEnvironment():
    def __init__(self,
                 pole_m=0.5,
                 cart_m=0.5,
                 pole_l=0.6,
                 mu=0.1,
                 g=9.82,

                 x_min=-6.0,
                 x_max=6.0,
                 x_dot_min=-10.0,
                 x_dot_max=10.0,
                 th_min=-np.pi,
                 th_max=np.pi,
                 th_dot_min=-10.0,
                 th_dot_max=10.0,
                 f_min=-10.0,
                 f_max=10.0,

                 target_x=0,
                 target_th=0,
                 target_margin=0.05,
                 target_time=1,

                 max_time=30,
                 sim_inter=0.01,
                 action_inter=0.1,
                 kinematics_integrator=None,
                 exit_on_target_time_reached=False):

        # model parameters from Deisenroth 2010
        self.pole_m = pole_m #mass pole
        self.cart_m = cart_m #mass cart
        self.pole_l = pole_l #length pole
        self.total_mass = pole_m + cart_m #total mass of the cart
        self.polemass_length = pole_m * pole_l #the pole mass length
        self.mu = mu #friction
        self.g = g #graviational pull

        # ranges given by project description, used for clipping
        self.x_lim = np.array([x_min, x_max]) #arrrey for min and max of x pos on x-Dim
        self.x_dot_lim = np.array([x_dot_min, x_dot_max]) #arrrey for min and max for x velocity
        self.th_lim = np.array([th_min, th_max]) #arrrey for min and max of theta angle
        self.th_dot_lim = np.array([th_dot_min, th_dot_max]) #arrrey for min and max of theta angular velocity
        self.f_lim = np.array([f_min, f_max]) #arrrey for min and max of force applied to cart

        # simulation and action intervals
        self.dt = sim_inter #time betwen each iteration
        self.dt_sq = sim_inter**2
        self.action_inter = action_inter #time between actions
        self.sim_steps = int(action_inter/sim_inter) #number df simulation steps between each action

        self.target = np.array([target_x, np.sin(target_th), np.cos(target_th)]) #target we want the pole to be in
        self.range = np.array([x_max-x_min, 2, 2]) #target tolerance we want the pole to be in
        self.target_margin = target_margin #target margin
        self.target_time = target_time #time how long the up state is reached

        self.target_reached = False #has pole reached the target?
        self.target_reached_temp = False #has the pole reached taret for first time?

        self.max_time = max_time #max trining/run time

        self.out_of_bounds = False #it the cart out of bounds

        self.done = False #is the learning task finisched?
        self.exit_on_target_time_reached = exit_on_target_time_reached #should the agent stop if target reached or exit?

        self.kinematics_integrator = kinematics_integrator #which kinematic integrater should be used?
                                                           # this can be initialized with "euler" or "none"

    def get_state(self): #state getter
        return self.state

    def init_state(self): # debut state printer
        print(f"init state: x: {self.state[0]}, x_dot: {self.state[1]}, th: {self.state[2]}, th_dot: {self.state[3]}")

    def reset(self): # reset the env
        self.state = np.array([0.0, 0.0, np.pi, 0.0])   # [x, x_dot, theta, theta_dot] inital state
        #self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time_taken = 0.0 # time taken to reach up position, set to 0
        self.time_close_to_target = 0.0 # time spent in upright position
        self.done = False #is the task finished?

        return self.state

    # function to performe the action step
    def action_step(self, force, noise=0):
        """
        The ODE is given by Deisenroth, who has a different angle definition than the lecture does
        (definition of x is the same).
        We use the ODE as provided by Deisenroth, hence we need to transform our angle theta and
        theta_dot before handing it to the ODE as follows:
        Transformation: theta_Deisenroth = np.pi - theta
                        theta_dot_Deisenroth = -theta
        theta_dot only occurs in squared form, hence no need to change sign
        After the calculation of the ODE we have to transform the angular acceleration back to our
        angle definition.
        All further calculations (theta_dot, theta) can then be performed using our angle definition.
        """
        # precalculate sin and cos to reduce computations, including transformation to Deisenroth angle definition
        force = np.clip(force, self.f_lim[0], self.f_lim[1]) #clip the force to lim
        force = force + noise #add noise

        x, x_dot, th, th_dot = self.state #unpack the state

        # run loop for nr of sim steps
        for i in range(self.sim_steps):
            # precals sin and cos in advance
            c_th = -np.cos(th)
            #c_th = np.cos(th)
            s_th = np.sin(th)

            # solve ODE
            # calc x dot dot
            x_dotdot = (2 * self.polemass_length * th_dot ** 2 * s_th + 3 * self.pole_m * self.g * s_th * c_th +
                        4 * force - 4 * self.mu * x_dot) / (4 * self.total_mass - 3 * self.pole_m * c_th ** 2)
            # calc theta dot dot
            th_dotdot = (-3 * self.polemass_length * th_dot ** 2 * s_th * c_th - 6 * self.total_mass * self.g * s_th -
                         6 * c_th * (force - self.mu * x_dot)) / (4 * self.pole_l * self.total_mass -
                                                                  3 * self.polemass_length * c_th ** 2)

            # transform theta_dotdot back to our angle definition
            th_dotdot = -th_dotdot

            # use 2 different integrators
            if self.kinematics_integrator == 'euler':  # explicit euler
                x = np.clip((x + x_dot * self.dt + 0.5 * x_dotdot * self.dt_sq), self.x_lim[0], self.x_lim[1])
                x_dot = np.clip((x_dot + x_dotdot * self.dt), self.x_dot_lim[0], self.x_dot_lim[1])
                th = th + th_dot * self.dt + 0.5 * th_dotdot * self.dt_sq
                th = th + ((th + np.pi) // (2 * np.pi)) * (-2 * np.pi)
                th_dot = np.clip((th_dot + th_dotdot * self.dt), self.th_dot_lim[0], self.th_dot_lim[1])
            else:
                # determine state variables using semi-implicit Euler's method, self.state = [x, x_dot, theta, theta_dot]
                x_dot = np.clip((x_dot + x_dotdot * self.dt), self.x_dot_lim[0], self.x_dot_lim[1])
                th_dot = np.clip((th_dot + th_dotdot * self.dt), self.th_dot_lim[0], self.th_dot_lim[1])
                x = np.clip((x + x_dot * self.dt + 0.5 * x_dotdot * self.dt_sq), self.x_lim[0], self.x_lim[1])
                th = th + th_dot * self.dt + 0.5 * th_dotdot * self.dt_sq
                th = th + ((th + np.pi) // (2 * np.pi)) * (-2 * np.pi)

                # todo fo we really need this code?
                # if x required clipping, warn user and set out_of_bounds-flag to True
                # if self.state[0] == self.x_lim[0] or self.state[0] == self.x_lim[1]:
                #   logging.warning(f"Cart reached limit in x-direction, state: {self.state}")
                #  self.out_of_bounds = True
                # todo: do we need to stop and reset the training if this happens? If so, use out_of_bounds-flag no because
                #  it never happens, because its clipped
                # could set force and velocity to 0 or below, following lines is for right boundary, analogous for left
                # F = min(F, 0)
                # s[1] = min(s[1], 0)

            # print(f"x: {x}, x_dot: {x_dot}, x_dotdot: {x_dotdot}, th: {th}, th_dot: {th_dot}, th_dotdot: {th_dotdot}")

        self.state = np.array([x, x_dot, th, th_dot]) # pack state

        self.time_taken += self.action_inter #add how much time has passed

        self.evaluate_target() #check if we are done yet

        return self.time_taken, self.state, self._calculate_reward(), self._done(self.state[0]) # self.out_of_bounds

    # functtion to check if the task is completed yet
    def _done(self, x):
        if not self.exit_on_target_time_reached:
            return x <= self.x_lim[0] or x >= self.x_lim[1] or self.time_taken >= self.max_time
        if self.exit_on_target_time_reached:
            return x <= self.x_lim[0] or x >= self.x_lim[1] or self.time_taken >= self.max_time or self.done

    # function to calc the reward according to slides
    def _calculate_reward(self):

        # for comparison: self.state = [x, x_dot, theta, theta_dot]

        """
        l = self.pole_l5
        A = 1
        T = A * np.array([[1, l, 0], [l, l ** 2, 0], [0, 0, l ** 2]])
        j = np.array([self.state[0], np.sin(self.state[2]), np.cos(self.state[2])])
        j_target = np.array([0, 0, 1])
        matmul = -0.5 * np.matmul(np.matmul((j - j_target), T), np.transpose(j - j_target))
        """

        matmul = (self.state[0] ** 2 + 2 * self.pole_l ** 2 * (1 - np.cos(self.state[2])) + 2 * self.pole_l *
                  self.state[0] * np.sin(self.state[2]))
        reward = -(1-np.exp(-0.5*matmul))
        #print(reward)
        return reward

    # fn to see if task is done yet by calculating absolute distance from current location and target
    # using the reward calculation's location definition: target = [x, sin(theta), cos(theta)]
    def evaluate_target(self):
        # how far is the pole from target away
        target_deviation = np.abs([self.state[0], np.sin(self.state[2]), np.cos(self.state[2])]-self.target)

        # how close are we to target
        # bool self.close_to_target is True if all variables <= target_margin
        self.close_to_target = np.all(target_deviation/self.range <= self.target_margin)

        # if close to target set vars
        if self.close_to_target:
            if not self.target_reached:
                self.target_reached = True
                self.target_reached_temp = True
                if not self.done:
                    self.time_close_to_target = self.time_taken
            else:
                # check that agent can hold cart in position
                if self.time_taken - self.time_close_to_target >= self.target_time and not self.done:
                    self.done = True

        elif not self.target_reached:
            self.target_reached = False
            self.time_close_to_target = 0.0


    # function to render and create video.
    # adapted from: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    def render(self, trajectory, name="video"):
        # get trajectory to right dims
        trajectory[:, 2] = np.pi - trajectory[:, 2]
        trajectory[:, 3] = -trajectory[:, 3]

        # init writer
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='TUM'), bitrate=-1)
        # create fig and plot
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111, aspect='equal', xlim=(-3.5, 3.5), ylim=(-0.7, 0.7))
        ax.grid()
        ax.get_xaxis().set_ticks([-6, 0, 6])
        ax.get_yaxis().set_ticks([0])

        # def how cart and pole look and add time
        pole, = ax.plot([], [], '-', linewidth=2, c='r')
        cart, = ax.plot([], [], 's', markersize=6, c='g')
        time_template = '%.1f s'
        time_text = ax.text(-6, 1.5, '')

        # anim init fn
        def init():
            pole.set_data([], [])
            cart.set_data([], [])
            time_text.set_text('')
            return pole, cart, time_text

        # actial animete fn
        def animate(i):
            x_cart = trajectory[i, 0]
            y_cart = 0
            x_pole = np.sin(trajectory[i, 2]) * self.pole_l + x_cart
            y_pole = - np.cos(trajectory[i, 2]) * self.pole_l
            cart.set_data([x_cart], [y_cart])
            pole.set_data([x_cart, x_pole], [y_cart, y_pole])
            time_text.set_text(time_template % (i * self.action_inter))
            return pole, cart, time_text

        # run the animation and save vid
        ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=self.action_inter, blit=True, init_func=init)
        ani.save(name + ".mp4", writer=writer, dpi=300)