
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

STATE_H = 80
STATE_W = 96
STATE_FRAME_CNT = 4
# discrete actions used by the agent
action_steer = [-1.0, -0.5, 0.0, 0.5, 1.0]
action_gas = [0.0, 0.5, 1.0]
action_break = [0.0, 0.5, 1.0]

BATCH_SIZE = 1

class EnvWrapper:
    state_queue = deque(maxlen=STATE_FRAME_CNT) # the state is made up of 4 most recent frames

    def set_initial_state(self):
        state = np.zeros([STATE_H, STATE_W], dtype=np.float32)

    def add_frame_to_state(self, frame):
        state_queue.append(frame)

    def get_state(self):
        state = np.asarray(self.state_queue)
        assert state.shape==(STATE_H, STATE_W, STATE_FRAME_CNT), state.shape
        return state


class Agent:

    def __init__():
        self.NN = NN()
        self.last_action, last_state = None, None

    def act(self, state, last_reward, done, episode):
        # first write an agent that only looks at the immediate reward (i.e. discount factor = 0)
        # so we only care about the current state, current reward

        actions_to_take, actions_to_take_idx = self.NN.forward_and_sample(state)

        if self.last_state is None: # Initialization phase, no training involved
            self.last_state = state
            self.last_action = actions_to_take_idx
            return actions_to_take

        self.NN.backward(*helper_make_batch_from_one(last_state, last_action, last_reward))

        # bookkeeping
        self.last_state = state
        self.last_action = actions_to_take_idx

        return actions_to_take

    def helper_make_batch_from_one(last_state, last_action, last_reward):
        return [np.expand_dims(last_state,0), np.expand_dims(last_action,0), np.expand_dims(last_reward,0)]




class NN:
    def __init__(self, gamma, session): 
        # Note for the gamma parameter: current impl implies gamma = 0, but the correct loss formula has gamma^t * reward
        # so I will just ignore this inconsistency between code and formula, and just use reward instead of gamma^t * reward
        self._sess = session

        # create placeholders
        self.true_action_steer = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)
        self.true_action_gas = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)
        self.true_action_break = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)
        # Note for the true_xxx placeholder: I think the true label is always the predicted action
        # by the definition of our loss function
        self.state = tf.placeholder(shape=[BATCH_SIZE, STATE_H, STATE_W, STATE_FRAME_CNT], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32)

        # build the policy network
        cur = self.state
        cur = self.conv_layer(cur, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4)
        cur = self.conv_layer(cur, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2)
        cur = self.fc_layer(cur, n_out=256)
        cur = tf.nn.relu(cur)
        logit_action_steer = self.fc_layer(cur, n_out=len(action_steer))
        logit_action_gas = self.fc_layer(cur, n_out=len(action_gas))
        logit_action_break = self.fc_layer(cur, n_out=len(action_break))

        self.prob_action_steer = tf.nn.softmax(logit_action_steer)
        self.prob_action_gas = tf.nn.softmax(logit_action_gas)
        self.prob_action_break = tf.nn.softmax(logit_action_break)

        # computing a lot of losses
        loss_action_steer = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_steer, self.true_action_steer)
        loss_action_gas = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_gas, self.true_action_gas)
        loss_action_break = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_break, self.true_action_break)

        loss = loss_action_steer + loss_action_gas + loss_action_break
        self.loss = self.reward * loss

        # some other stuffs, and variable initialization
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        self._sess.run(tf.initialize_all_variables())


    def conv_layer(self, input, n_in_channel, n_out_channel, filter_size, stride, hasRelu=True):
        # TODO conv layer without adding bias ( bias is not used in paper either). but I could try it later if time permitted. tf.nn.bias_add
        filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_in_channel, n_out_channel], stddev=.01))
        output = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding='SAME')
        # output = _instance_norm(output) # TODO maybe batch normalization here to avoid bad initialization problem
        if hasRelu:
            output = tf.nn.relu(output)
        print("conv layer, output size: %s" % ([i.value for i in output.get_shape()]))
        return output

    def fc_layer(self, input, n_out):
        n_in = np.prod(input.get_shape().as_list()[1:])
        input = tf.reshape(input, [-1, n_in])

        W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01))
        bias = tf.Variable(tf.truncated_normal([n_out], stddev=0.01))
        output = tf.nn.bias_add(tf.matmul(input, W), bias)
        print ("fc layer, input*output : %d*%d" % (n_in, n_out))
        return output

    def forward_and_sample(self, input):
        probs = self._sess.run([self.prob_action_steer, self.prob_action_gas, self.prob_action_break], 
            feed_dict={self.state=input})
        steerIDX = np.random.choice(len(action_steer), p=probs[0])
        gasIDX = np.random.choice(len(action_gas), p=probs[1])
        breakIDX = np.random.choice(len(action_break), p=probs[2])
        return [action_steer[steerIDX], action_gas[gasIDX], action_break[breakIDX]], [steerIDX, gasIDX, breakIDX]

    def backward(self, state_batch, action_batch, reward_batch):
        self._sess.run(self.train_op, feed_dict={
                self.state:             state_batch,
                self.true_action_steer: action_batch[:, 0],
                self.true_action_gas:   action_batch[:, 1],
                self.true_action_break: action_batch[:, 2],
                self.reward:            reward_batch,
            })


if __name__=='__main__':
    pass

