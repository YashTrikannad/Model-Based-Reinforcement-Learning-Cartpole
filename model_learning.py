# ==========================================
# Title:  Model Learning Code
# Author: Ashish Mehta
# Date:   20 October 2018
# ==========================================

import tensorflow as tf
import numpy as np
import gym


class Model:
    """
    Learns a model of an environment using a DNN for function approximation.
    Data is collected using random policies and stochastic noise policies.
    """
    def __init__(self, restore=False, restore_file="./tmp/model120000.ckpt"):
        """

        :param restore: [bool] Restores learned model from disk if True.
                        Initiates model weights randomly for training if False.
        :param restore_file: File adderess to be used to restore weights
        """
        self._env = gym.make('Pendulum-v0')

        self._num_hidden1 = 50
        self._num_hidden2 = 50
        self._learning_rate = 6e-4

        self._restore = restore
        self._restore_file = restore_file
        self.NN_model()

    def NN_model(self):
        """
        2-layer Fully connected NN
        """
        self.input_tensor = tf.placeholder(tf.float32, [None, 4], name="previous_state_action")

        W1 = tf.get_variable("W1", shape=[4, self._num_hidden1], initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.Variable(tf.zeros([self._num_hidden1]), name="B1")
        layer1 = tf.nn.relu(tf.matmul(self.input_tensor, W1) + B1)

        W2 = tf.get_variable("W2", shape=[self._num_hidden1, self._num_hidden2],
                             initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.Variable(tf.zeros([self._num_hidden1]), name="B2")
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + B2)

        W3 = tf.get_variable("W3", shape=[self._num_hidden2, 3], initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.Variable(tf.zeros([3]), name="B3")
        self.ground_truth = tf.placeholder(tf.float32, [None, 3], name="ground_truth")
        self.prediction = tf.matmul(layer2, W3) + B3

        self.loss_op = tf.reduce_mean(tf.square(self.ground_truth - self.prediction))
        tf.summary.scalar('MSE loss', self.loss_op)
        self.n_step_loss_stochastic = tf.Variable(0.0)
        tf.summary.scalar('MSE_n-step_loss_stochastic', self.n_step_loss_stochastic)
        self.n_step_loss_random = tf.Variable(0.0)
        tf.summary.scalar('MSE_n-step_loss_random', self.n_step_loss_random)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()

        self.sess = tf.Session()
        if self._restore:
            self.saver.restore(self.sess, self._restore_file)
            print("Model restored successfully. ")
        else:
            self.sess.run(self.init)
            print("Model initialized. ")

        self.train_writer = tf.summary.FileWriter('./temp/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter('./temp/val')

    @staticmethod
    def write_to_disk(data, filename):
        """
        Writes the given data to disk as a CSV (human readable)
        :param data: data array to be written to file
        :param filename: file name to store the data
        """
        file = open(filename + '.csv', 'a')
        comma = ','
        for x in np.nditer(data):
            file.write(str(x))
            file.write(comma)
        file.write('\n')
        file.close()

    @staticmethod
    def read_from_disk(filename):
        """
        Read data from the disk as a 2D array
        :param filename: File name to read data from
        :return: 2D array of read data
        """
        data_points = []
        file = open(filename + '.csv', 'r')
        while True:
            f = file.readline().split(',')
            data = [float(i) for i in f[:-1]]
            if data:
                data_points.append(data)
            else:
                break
        file.close()
        return np.array(data_points)

    def random_sampler(self, num_samples):
        """
        Randomly sample trajectories and store them to the disk
        :param num_samples: total number of samples to sample
        """
        prev_observation = self._env.reset()
        for t in range(num_samples):
            # Uncomment if you want to render
            # self._env.render()
            action = self._env.action_space.sample()
            next_observation, reward, done, info = self._env.step(action)
            data = np.array([prev_observation[0], prev_observation[1], prev_observation[2], action,
                             next_observation[0], next_observation[1], next_observation[2], reward,
                             done])  # s(t), a(t), s(t+1), r, d
            self.write_to_disk(data, 'data')
            prev_observation = next_observation
            if t % 3 == 0:
                prev_observation = self._env.reset()
                print("Episode finished after {} timesteps".format(t + 1))
        print('Random data generated')

    def temporal_noise_sampler(self, num_samples):
        """
        Randomly sample trajectories using temporal noisy actions and store them to the disk
        :param num_samples: total number of samples to sample
        """
        prev_observation = self._env.reset()

        # random sample of action value and action duration sampled from a uniform distribution
        u_noise_val = np.random.uniform(-2, 2)
        u_noise_time = np.random.randint(5, 35)

        # time counter for particular noise values
        noise_t = 0
        for t in range(num_samples):
            noise_t = noise_t + 1

            # Uncomment if you want to render
            # self._env.render()

            action = u_noise_val
            next_observation, reward, done, info = self._env.step([action])
            data = np.array([prev_observation[0], prev_observation[1], prev_observation[2], action,
                             next_observation[0], next_observation[1], next_observation[2], reward,
                             done])  # [s(t), a(t), s(t+1), reward, done]
            self.write_to_disk(data, 'data')
            prev_observation = next_observation

            # reinitialize noise params after u_noise_time time steps
            if noise_t == u_noise_time:
                print("Action {} changed after {} timesteps".format(u_noise_val, u_noise_time))
                noise_t = 0
                u_noise_val = np.random.uniform(-2, 2)
                u_noise_time = np.random.randint(5, 35)

            # reset the environment after every 500 timesteps
            if t % 500 == 0:
                prev_observation = self._env.reset()
                noise_t = 0
                u_noise_val = np.random.uniform(-2, 2)
                u_noise_time = np.random.randint(5, 30)
                print("Episode finished after {} timesteps \n".format(t + 1))

        print('Temporal data generated')

    @staticmethod
    def dataset_generator(data_points, batch_size):
        """
        Generates a dataset which can be used for either training or validation
        :param data_points: A 2D array of data used to generate the dataset
        :param batch_size: [int] minibatch size of the dataset generated
        :yield: [X_batch, Y_batch, current_epoch]:  A minibatch of data X, corresponding minibatch of data Y
                                                    and current epoch
        """

        current_epoch = 0
        while True:
            current_epoch = current_epoch + 1
            # random shuffle data
            np.random.shuffle(data_points)

            X = data_points[:, 0:4]  # Input data => [cos(theta_t), sin(theta_t), theta_dot_t, action]
            Y = data_points[:, 4:7] - data_points[:, 0:3]  # Expected output =>
            # [cos(theta_t+1) - cos(theta_t), sin(theta_t+1) - sin(theta_t), theta_dot_t+1 - theta_dot_t]

            for batch in range(X.shape[0] // batch_size):
                X_batch = X[batch * batch_size: batch * batch_size + batch_size, :]
                Y_batch = Y[batch * batch_size: batch * batch_size + batch_size, :]

                yield X_batch, Y_batch, current_epoch

    def train_model(self, num_epochs, train_dataset, validation_dataset):
        """
        Trains a model using train_dataset and validates it on unseen validation_dataset.
        The function also computes n_step_loss every 100 steps and saves the model every 10000 steps.
        :param num_epochs: [int] Number of epochs to train the model
        :param train_dataset: [self.dataset_generator] A dataset_generator which yields the training data
        :param validation_dataset: [self.dataset_generator A dataset_generator which yields the validation_dataset
        """
        epoch = 1
        step = 0
        while epoch <= num_epochs:
            step = step + 1

            # Yield training data
            X, Y, epoch = next(train_dataset)

            # Train NN and write summary
            _, loss, pred, summary = self.sess.run([self.train_op, self.loss_op, self.prediction, self.merged],
                                     feed_dict={self.input_tensor: X, self.ground_truth: Y})
            self.train_writer.add_summary(summary, step)

            if step % 100 == 0:

                #Compute N_step losses
                stochastic_nstep_loss, _, _ = self.n_step_prediction_loss(50, 'stochastic_noise')
                self.n_step_loss_stochastic.load(stochastic_nstep_loss, self.sess)

                random_nstep_loss, _, _ = self.n_step_prediction_loss(50, 'random')
                self.n_step_loss_random.load(random_nstep_loss, self.sess)

                # print("Step " + str(step) + ", Minibatch Loss= " + \
                #       "{:.4f}".format(loss) + "\n Prediction= " + str(pred) + " \n Ground truth= " + str(
                #     Y) + "\n\n")

                # Validate model and write validation summary
                X_val, Y_val, _ = next(validation_dataset)
                val_loss, val_pred, val_summary = self.sess.run([self.loss_op, self.prediction, self.merged],
                                                       feed_dict={self.input_tensor: X_val,
                                                                  self.ground_truth: Y_val})
                self.val_writer.add_summary(val_summary, step/100)

            # Save model to disk
            if step % 10000 == 0:
                save_path = self.saver.save(self.sess, "./tmp/model{}.ckpt".format(step))
                print("Step ", step)
                print("Model saved to disk. ")

        print("Model trained")

    def predict_using_model(self, input_vec):
        """
        Function to predict the next state using the NN model, given the current state and action
        :param input_vec: [n, 4] 2D array containing n [cos(theta_t), sin(theta_t), thetadot_t, action_t]
        :return: 2D array containing n [cos(theta_t+1), sin(theta_t+1), thetadot_t+1]
        """
        pred = self.sess.run(self.prediction, feed_dict={self.input_tensor: input_vec})
        next_state = pred + input_vec[:, 0:3]
        return next_state

    def n_step_prediction_loss(self, n_steps=50, sampling_policy='stochastic_noise'):
        """
        Samples new trajectories from the simulator using the policy provided by sampling_policy.
        Computes loss after N-step for these trajectories.
        :param n_steps: Number of future steps to compute n-step loss
        :param sampling_policy: [String] 'stochastic_noise' or 'random' policy for collecting data.
        :return: n_step loss, 2D array of  simulator observations, 2D array of  model predictions
        """

        prev_observation_simulator = self._env.reset()
        prev_observation_model = prev_observation_simulator
        next_observation_model = []
        next_observation_simulator = []

        # random sample of action value and action duration sampled from a uniform distribution
        u_noise_val = np.random.uniform(-2, 2)
        u_noise_time = np.random.randint(5, 35)

        noise_t = 0
        for t in range(n_steps):
            noise_t = noise_t + 1

            # Uncomment if you want to render
            # self._env.render()

            # Choose action using sampling policy
            if sampling_policy == 'stochastic_noise':
                action = u_noise_val
            elif sampling_policy == 'random':
                action = self._env.action_space.sample()[0]

            # Compute next states
            [model_prediction] = self.predict_using_model(np.array([[prev_observation_model[0],
                                                               prev_observation_model[1],
                                                               prev_observation_model[2], action]]))
            next_observation_model.append(model_prediction)
            ob, _, _, _ = self._env.step([action])
            next_observation_simulator.append(ob)

            prev_observation_model = next_observation_model[-1]

            # reinitialize noise params after u_noise_time time steps
            if noise_t == u_noise_time:
                # print("Action {} changed after {} timesteps".format(u_noise_val, u_noise_time))
                noise_t = 0
                u_noise_val = np.random.uniform(-2, 2)
                u_noise_time = np.random.randint(5, 35)

        # compute MSE nstep loss
        loss = np.mean(np.square(next_observation_simulator[-1] - next_observation_model[-1]))
        return loss, np.array(next_observation_simulator), np.array(next_observation_model)


if __name__=='__main__':
    M=Model(True)
    for i in range(1):
        print(M.n_step_prediction_loss(50, 'random'))