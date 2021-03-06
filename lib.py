from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pprint as pp, heapq, math
from IPython import embed
import ipdb
import helper

STATE_H = 80
STATE_W = 96
STATE_FRAME_CNT = 4
# discrete actions used by the agent
action_steer = [-1.0, 0.0, 1.0]
action_gas = [0.0, 1.0]
action_break = [0.0, 0.8]
def map_to_discrete(action_vec):
    steerR, gasR, breakR = action_vec
    steer_dis = [i for i,val in enumerate(action_steer) if abs(val-steerR)<1e-3][0]
    gas_dis = [i for i,val in enumerate(action_gas) if abs(val-gasR)<1e-3][0]
    break_dis = [i for i,val in enumerate(action_break) if abs(val-breakR)<1e-3][0]
    return steer_dis, gas_dis, break_dis


BATCH_SIZE = 5
GAMMA = 0.999

class EnvHelper:
    state_queue = deque(maxlen=STATE_FRAME_CNT) # the state is made up of 4 most recent frames

    def add_frame_to_state(self, frame):
        self.state_queue.append(frame)

    def get_state(self):
        assert len(self.state_queue) == STATE_FRAME_CNT, "make sure the queue is full through proper initialization, before trying to extract state from it"
        state = np.transpose(np.asarray(self.state_queue), (1,2,0)) # np.transpose() permutes axes
        assert state.shape==(STATE_H, STATE_W, STATE_FRAME_CNT), state.shape
        return state


class Agent:

    def __init__(self):
        self._sess = tf.Session()
        self.NN = NN(self._sess, self)
        self.NNb = NNForBaseline(self._sess, self)
        self.stepsStat = helper.RunningPercentile(0.9)
        self.stepsAlive = 0 # bookkeeping the step count of this episode

        self.last_action, self.last_state = None, None
        self.init_batch()
        self.debug = False
        self.startPolicyTraining = True

    def init_batch(self):
        self.epEnd = set() # stores the time step where the episode ends
        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []

    def act(self, state, last_reward, done, epCnt):
        actions_to_take, actions_to_take_idx = self.NN.forward_and_sample(state)

        if self.last_state is None: # Initialization phase, no training involved
            self.last_state = state
            self.last_action = actions_to_take_idx
            self.stepsAlive += 1
            return actions_to_take

        self.state_batch.append(self.last_state)
        self.action_batch.append(self.last_action)

        if done:
            self.stepsStat.add(self.stepsAlive)
            self.reward_batch.append(1.0 if self.stepsAlive > self.stepsStat.get() else -1.0) # calculate my internal reward
            assert abs(self.reward_batch[-1]) == 1.0 # make sure epEnd and reward_batch is correctly aligned
            self.epEnd.add(len(self.reward_batch)-1)
            print 'episode: {} step: {}  perc,size: {},{} reward: {:.1f}'.format(epCnt,self.stepsAlive,self.stepsStat.get(), self.stepsStat.len(),self.reward_batch[-1])
            self.stepsAlive = 0
            last_state = None
        else:
            self.reward_batch.append(0.0) # calculate my internal reward

        if len(self.epEnd) == BATCH_SIZE: # a batch of episodes has been collected
            discounted_r = self.discount_rewards(self.reward_batch, self.epEnd)
            if self.debug: 
                self.debug=False
                printnorm(self._sess)
                embed()
            r_minus_base = discounted_r - self.NNb.forward(self.state_batch).ravel()
            self.NNb.backward(self.state_batch, discounted_r)
            if self.startPolicyTraining:
                self.NN.backward(self.state_batch, self.action_batch, r_minus_base)
            self.init_batch() # clean up

        # bookkeeping
        self.last_state = state
        self.last_action = actions_to_take_idx
        self.stepsAlive += 1

        return actions_to_take


    def discount_rewards(self, r, epEnd):
        """ take 1D float array of rewards and compute discounted reward """
        res = np.zeros_like(r)
        running_add = 0.0
        assert (len(r)-1) in epEnd, 'simple sanity check: the last reward must be the episode end'
        assert isinstance(r, list)
        for t in reversed(xrange(len(r))):
            if t in epEnd: running_add = 0.0 # reset 
            running_add = GAMMA * running_add + r[t]
            res[t] = running_add
        return res

class NN:
    def __init__(self, sess, ref_to_agent, lr=0.0001): 
        self._sess = sess
        self.debug = False
        self.printAct = False
        self.agent = ref_to_agent

        # create placeholders
        self.true_action_steer = tf.placeholder(shape=[None], dtype=tf.int32)
        self.true_action_gas = tf.placeholder(shape=[None], dtype=tf.int32)
        self.true_action_break = tf.placeholder(shape=[None], dtype=tf.int32)
        # Note for the true_xxx placeholder: I think the true label is always the predicted action
        # by the definition of our loss function
        self.state = tf.placeholder(shape=[None, STATE_H, STATE_W, STATE_FRAME_CNT], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)

        with tf.variable_scope('po'): # build the policy network
            self.conv1 = conv_layer(self.state, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4, name='conv1')
            self.conv2 = conv_layer(self.conv1, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2, name='conv2')
            self.fc1, self.fc1W = fc_layer(self.conv2, n_out=256, name='fc1')
            self.relu1 = tf.nn.relu(self.fc1)
            self.relu1_d = tf.nn.dropout(self.relu1, 0.5)
            self.logit_steer, self.steerW = fc_layer(self.relu1_d, n_out=len(action_steer), name='fc_steer')
            self.logit_gas, self.gasW = fc_layer(self.relu1_d, n_out=len(action_gas), name='fc_gas')
            self.logit_break,self.breakW = fc_layer(self.relu1_d, n_out=len(action_break), name='fc_break')

            self.prob_action_steer = tf.nn.softmax(self.logit_steer)
            self.prob_action_gas = tf.nn.softmax(self.logit_gas)
            self.prob_action_break = tf.nn.softmax(self.logit_break)

            # computing a lot of losses
            self.loss_action_steer = tf.reduce_mean(self.reward * tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logit_steer, self.true_action_steer))
            self.loss_action_gas = tf.reduce_mean(self.reward * tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logit_gas, self.true_action_gas))
            self.loss_action_break = tf.reduce_mean(self.reward * tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logit_break, self.true_action_break))
            self.reg_loss = 0.00 * tf.nn.l2_loss(self.fc1W)

            self.loss = self.loss_action_steer + self.loss_action_gas + self.loss_action_break + self.reg_loss

            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        self._sess.run(tf.initialize_variables( tf.get_collection(tf.GraphKeys.VARIABLES, scope='po') ))

    def forward_and_sample(self, input):
        input = np.expand_dims(input,0) # make a batch
        probs = self._sess.run([self.prob_action_steer, self.prob_action_gas, self.prob_action_break], 
            feed_dict={self.state:input})

        assert probs[0].shape == (1, len(action_steer))
        steerIDX = np.random.choice(len(action_steer), p=probs[0][0])
        gasIDX = np.random.choice(len(action_gas), p=probs[1][0])
        breakIDX = np.random.choice(len(action_break), p=probs[2][0])
        a = [action_steer[steerIDX], action_gas[gasIDX], action_break[breakIDX]]
        if self.printAct:
            logits = self._sess.run([self.logit_steer, self.logit_gas, self.logit_break], feed_dict={self.state:input})
            pp.pprint([[v for v in act_logit[0]] for act_logit in logits] + [a])
        if self.debug:
            self.debug = False
            [conv1,conv2,fc1,fc_steer,fc_gas,fc_break] = self._sess.run([self.conv1, self.conv2, self.fc1, self.logit_steer, self.logit_gas, self.logit_break], feed_dict={self.state:input})
            embed()
        return a, [steerIDX, gasIDX, breakIDX]

    def backward(self, state_batch, action_batch, reward_batch):
        action_batch = np.asarray(action_batch, dtype=np.int32)
        _, loss, loss_steer, loss_gas, loss_break = self._sess.run(
            [self.train_op, self.loss, self.loss_action_steer, self.loss_action_gas, self.loss_action_break], feed_dict={
                self.state:             state_batch,
                self.true_action_steer: action_batch[:, 0],
                self.true_action_gas:   action_batch[:, 1],
                self.true_action_break: action_batch[:, 2],
                self.reward:            reward_batch,
            })
        print ('NN  loss: %f loss_steer: %f loss_gas: %f loss_break: %f' % (loss, loss_steer, loss_gas, loss_break) )

class NNForBaseline:
    def __init__(self,sess,ref_to_agent):
        self._sess = sess
        self.debug = False
        self.printV = False
        self.agent = ref_to_agent

        # create placeholders
        self.state = tf.placeholder(shape=[None, STATE_H, STATE_W, STATE_FRAME_CNT], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)

        
        with tf.variable_scope('ba'): # build the value network
            self.conv1 = conv_layer(self.state, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4, name='conv1')
            self.conv2 = conv_layer(self.conv1, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2, name='conv2')
            self.fc1, self.fc1W = fc_layer(self.conv2, n_out=256, name='fc1')
            self.relu1 = tf.nn.relu(self.fc1)
            self.relu1_d = tf.nn.dropout(self.relu1, 0.5)
            self.value, self.valueW = fc_layer(self.relu1_d, n_out=1, name='value',bias_init=tf.constant([0.0]))
            self.value = tf.reshape(self.value,[-1]) # if you don't flatten it, you will get a n*n matrix when you do self.value-self.reward

            self.reg_loss = 0.5 * tf.nn.l2_loss(self.fc1W)
            self.loss = tf.reduce_mean(tf.square(self.value - self.reward)) + self.reg_loss

            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        self._sess.run(tf.initialize_variables( tf.get_collection(tf.GraphKeys.VARIABLES, scope='ba') ))

    def forward(self, input):
        value = self._sess.run(self.value, feed_dict={self.state: input})
        if self.printV:
            print 'value:' + str(value)
        if self.debug:
            self.debug = False
            [conv1,conv2,fc1] = self._sess.run([self.conv1, self.conv2, self.fc1], feed_dict={self.state:input})
            embed()
        return value

    def backward(self, state_batch, reward_batch):
        _, loss, reg_loss = self._sess.run([self.train_op, self.loss, self.reg_loss],
                            feed_dict={self.state: state_batch, self.reward: reward_batch})
        print 'NNb loss: {:f} reg_loss: {:f}'.format(loss, reg_loss)



def conv_layer(input, n_in_channel, n_out_channel, filter_size, stride, name, hasRelu=True):
    with tf.variable_scope(name):
        filt = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_in_channel, n_out_channel], stddev=.001), name='W')
        bias = tf.Variable(tf.constant(1.0,shape=[n_out_channel]), name='b')
    output = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding='SAME')
    output = tf.nn.bias_add(output, bias)
    # output = _instance_norm(output) # TODO maybe batch normalization here to avoid bad initialization problem
    if hasRelu:
        output = tf.nn.relu(output)
    print("conv layer, output size: %s" % ([i.value for i in output.get_shape()]))
    return output

def fc_layer(input, n_out, name, bias_init=None):
    n_in = np.prod(input.get_shape().as_list()[1:])
    input = tf.reshape(input, [-1, n_in])
    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.001), name='W')
        bias = tf.Variable(tf.truncated_normal([n_out], stddev=0.001) if bias_init==None else bias_init, name='b')
    output = tf.nn.bias_add(tf.matmul(input, W), bias)
    print ("fc layer, input*output : %d*%d" % (n_in, n_out))
    return output, W

def plot(inputs):
    # example usage: plot([conv1, conv2]); plot([state])
    row, col = len(inputs), inputs[0].shape[3]
    f,ax=plt.subplots(row,col)
    ax = ax.reshape(row,col)
    for i in xrange(row):
        for j in xrange(col):
            ax[i][j].imshow(inputs[i][0,:,:,j])
    plt.show()

def printnorm(sess):
    normdict={}
    for v in tf.all_variables(): 
     if '/W:0' in v.name or '/b:0' in v.name:
         normdict[v.name]=np.linalg.norm(sess.run(v))
    pp.pprint(normdict)


if __name__=='__main__':
    pass

