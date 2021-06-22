import numpy as np
import tensorflow as tf

#class to initiate the NN
class clsNetwork(tf.keras.Model):
    def __init__(self, num_s, hidden_l, num_a):
        # def superclass for class inheritance
        super(clsNetwork, self).__init__()
        # def the tf input layer with the Nr of states
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_s,))
        # create array to iterate through the hiddenlayers
        self.hidden_l_array = []
        # def the initializer for the layers
        #initializer = tf.keras.initializers.GlorotNormal()
        initializer = tf.keras.initializers.RandomNormal()
        # iterate through the hidden layers
        for i in hidden_l:
            # def the hidden layers
            self.hidden_l_array.append(tf.keras.layers.Dense(i, activation='relu', use_bias=True,
                                                            bias_initializer='zeros', kernel_initializer=initializer))
        # def the output layers
        self.output_layer = tf.keras.layers.Dense(num_a, activation='linear', kernel_initializer=initializer)

    # specific tf function to call and run the network overrides the predefinded tf function
    @tf.function
    def call(self, input_l):
        # calc intermediate result after inputlayer
        z = self.input_layer(input_l)
        # iterate through the layers and calc the intermediate results
        for layer in self.hidden_l_array: z = layer(z)
        # cals the logits
        output = self.output_layer(z)
        return output

# class to define the Network
class clsDQN(tf.Module):
    def __init__(self, num_s, num_a, hidden_l, gamma, optimizer, experience_buffer_size, batch_size):
        # def super class for inheritance of class
        super(clsDQN, self).__init__()
        # init the NN of the model
        self.model = clsNetwork(num_s, hidden_l, num_a)
        # pass the optimizer
        self.optimizer = optimizer

        # discount factor gamma
        self.gamma = gamma
        # number of actions
        self.num_a = num_a
        # def of batch size
        self.batch_size = batch_size

        # def the experience buffer
        self.experience_buffer = {'s': [], 'a': [], 'r': [], 's_1': [], 'done': []}
        # pass the experience buffer size
        self.experience_buffer_size = experience_buffer_size

    # function to run the data through the model/NN
    def run_network(self, input_l):
        return self.model(tf.convert_to_tensor(input_l, tf.float32))

    # function to train the network paamters via gradient decent
    def train(self, QA2_net):
        # choose a random sample from the experience buffer
        rand_exp = np.random.randint(low=0, high=len(self.experience_buffer['s']), size=self.batch_size)

        # load the expereince from the experience buffer
        S = tf.convert_to_tensor([self.experience_buffer['s'][i] for i in rand_exp], tf.float32)
        A = tf.convert_to_tensor([self.experience_buffer['a'][i] for i in rand_exp], tf.float32)
        R = tf.convert_to_tensor([self.experience_buffer['r'][i] for i in rand_exp], tf.float32)
        S_1 = tf.convert_to_tensor([self.experience_buffer['s_1'][i] for i in rand_exp], tf.float32)
        DONE = tf.convert_to_tensor([self.experience_buffer['done'][i] for i in rand_exp], tf.bool)

        # calculate the q-value from the Q2 network
        q_star = tf.where(DONE, R, R + self.gamma * tf.math.reduce_max(QA2_net.run_network(S_1), axis=1))

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            # calculate the q-value from the Q1 network
            q_val = tf.math.reduce_sum(self.run_network(S) * tf.one_hot(tf.cast(A, tf.int32), self.num_a), axis=1)
            # calculate loss
            loss = tf.math.reduce_sum(tf.square(q_star - q_val))

        # tell tf to calc gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # tell tf to update values with optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    # function to get action by running states through the NN
    def get_action(self, S, epsilon):
        do = np.random.choice([0, 1], p=[1-epsilon, epsilon])
        if do == 0: return np.argmax(self.run_network(np.atleast_2d(S)))
        elif do == 1: return np.random.choice(self.num_a)

    # add a new experience to the experience buffer
    def store_experience(self, exp):
        if len(self.experience_buffer['s']) >= self.experience_buffer_size:
            for key in self.experience_buffer.keys():
                self.experience_buffer[key].pop(0)
        for key, value in exp.items():
            self.experience_buffer[key].append(value)

    # copy the weights from Q1 ro Q2
    def copy_Q1_model(self, QA1_net):
        Q_var = QA1_net.model.trainable_variables
        Q2_var = self.model.trainable_variables
        for Q2_var_temp, Q_var_temp in zip(Q2_var, Q_var):
            Q2_var_temp.assign(Q_var_temp.numpy())

    def update_Q2_model(self, QA1_net):
        Q_var = QA1_net.model.trainable_variables
        Q2_var = self.model.trainable_variables
