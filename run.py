
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

    def act(state, reward):
        # first write an agent that only looks at the immediate reward (i.e. discount factor = 0)
        # so we only care about the current state, current reward



class NN:
    def __init__(self, true_action_steer, true_action_gas, true_action_break, 
                    state, reward, gamma): 
        # Note for the true_xxx parameters: I think the true label is always the predicted action
        # by the definition of our loss function
        # Note for the gamma parameter: current impl implies gamma = 0, but the correct loss formula has gamma^t * reward
        # so I will just ignore this inconsistency between code and formula, and just use reward instead of gamma^t * reward

        cur = state
        cur = self.conv_layer(cur, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4)
        cur = self.conv_layer(cur, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2)
        cur = self.fc_layer(cur, n_out=256)
        cur = tf.nn.relu(cur)
        logit_action_steer = self.fc_layer(cur, n_out=len(action_steer))
        logit_action_gas = self.fc_layer(cur, n_out=len(action_gas))
        logit_action_break = self.fc_layer(cur, n_out=len(action_break))


        loss_action_steer = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_steer, true_action_steer)
        loss_action_gas = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_gas, true_action_gas)
        loss_action_break = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logit_action_break, true_action_break)


        loss = loss_action_steer + loss_action_gas + loss_action_break
        loss = reward * loss

    def conv_layer(self, input, n_in_channel, n_out_channel, filter_size, stride, hasRelu=True):
        # TODO conv layer without adding bias ( bias is not used in paper either). but I could try it later if time permitted. tf.nn.bias_add
        filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_in_channel, n_out_channel], stddev=.01))
        output = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding='SAME')
        # output = _instance_norm(output) # TODO read what is instance normalization 
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



if __name__=='__main__':
    pass

