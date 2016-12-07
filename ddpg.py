"""
Implementation of DDPG - Deep Deterministic Policy Gradient

I tried the following things, but still not working: (search for # tried_but_not_worked to locate them)

+ initialized the weights with my supervised-learning pretrained model at the beginning of training
+ fill the replay memory (10000 frames) with human play data at the beginning of training
+ I thought the negative reward -100 given when the agent drives out of the lane is around 2 orders of magnitude larger than the normal reward at each time step, which might incur a very large gradient and make the learning instable, so I changed it from -100 to -1. But still not work.
+ add exploration with Ornstein-Uhlenbeck process as suggested by the DDPG paper(seems unhelpful)

Even combining these, the ouput logit is still monotoncially increasing, making the steering action very close to 1.0,  and the car always turn right.
So I guess there is some bugs in the DDPG implementation
Original code is from https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py 
written by Patrick Emami
I changed the input and output and network archetecture.
And changed the mini-batch training frequency --- original frequency is 
training at every time step, resulting in overfitting to a poor policy (always turning left/right)
since the agent cannot see any nice states-action pairs due to its poor policy.
"""
import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np, gym, tflearn
from IPython import embed
from helper import printdebug
from replay_buffer import ReplayBuffer
import my_car_env, lib, time

# ==========================
#   Training Parameters
# ==========================
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.0001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001
STATE_H = 80
STATE_W = 96
STATE_FRAME_CNT = 4

# ===========================
#   Utility Parameters
# ===========================
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 200



def conv_layer(input, n_in_channel, n_out_channel, filter_size, stride, name, hasRelu=True):
    n_in = np.prod(input.get_shape().as_list()[1:])
    with tf.variable_scope(name):
        initval = np.sqrt(3)/np.sqrt(n_in)
        filt = tf.Variable(tf.random_uniform([filter_size, filter_size, n_in_channel, n_out_channel],-initval,initval), name='W')
        bias = tf.Variable(tf.constant(0.0,shape=[n_out_channel]), name='b')
    output = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding='SAME')
    output = tf.nn.bias_add(output, bias)
    # output = _instance_norm(output) # TODO maybe batch normalization here to avoid bad initialization problem
    if hasRelu:
        output = tf.nn.relu(output)
    print("conv layer, output size: %s initval: %f" % ([i.value for i in output.get_shape()], initval))
    return output

def fc_layer(input, n_out, name, weight_init=None, bias_init=None):
    n_in = np.prod(input.get_shape().as_list()[1:])
    input = tf.reshape(input, [-1, n_in])
    with tf.variable_scope(name):
        # initval = np.sqrt(3)/np.sqrt(n_in)
        # W = tf.Variable(tf.random_uniform([n_in, n_out],-initval,initval) if weight_init==None else weight_init, name='W', dtype=tf.float32)
        W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.001) if weight_init==None else weight_init, name='W', dtype=tf.float32)
        bias = tf.Variable(tf.truncated_normal([n_out], stddev=0.001) if bias_init==None else bias_init, name='b')
    output = tf.nn.bias_add(tf.matmul(input, W), bias)
    print ("fc layer, input*output : %d*%d" % (n_in, n_out))
    return output, W


class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, action_dim, learning_rate, tau):
        self.sess = sess
        self.printLogits = False
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.steer_gas_break = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_steer_gas_break = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.steer_gas_break, self.network_params, -self.action_gradient)
        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=[None, STATE_H, STATE_W, STATE_FRAME_CNT])
        conv1 = conv_layer(inputs, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4, name='conv1')
        conv2 = conv_layer(conv1, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2, name='conv2')
        fc1, fc1W = fc_layer(conv2, n_out=256, name='fc1')
        relu1 = tf.nn.relu(fc1)
        relu1_d = tf.nn.dropout(relu1, 0.5)

        b_init = tf.constant([0,5,-5], dtype=tf.float32)
        w_init = tf.random_uniform([256,3],-0.0003,0.0003)
        fc_out, fc_outW = fc_layer(relu1_d, self.a_dim, 'fc_out', w_init, bias_init=b_init)
        steer_gas_break = tf.nn.tanh(fc_out)*[1.0,0.5,0.5]+[0.0,0.5,0.5]
        return inputs, fc_out, steer_gas_break

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        out_np, steer_gas_break_np = self.sess.run([self.out, self.steer_gas_break], feed_dict={
            self.inputs: inputs
        })
        if self.printLogits and inputs.shape[0] == 1:
            print ["%0.2f" % o for o in out_np[0]]
        return steer_gas_break_np

    def predict_target(self, inputs):
        return self.sess.run(self.target_steer_gas_break, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, STATE_H, STATE_W, STATE_FRAME_CNT])
        action = tflearn.input_data(shape=[None, self.a_dim])
        conv1 = conv_layer(inputs, n_in_channel=STATE_FRAME_CNT, n_out_channel=16, filter_size=8, stride=4, name='conv1')
        conv2 = conv_layer(conv1, n_in_channel=16, n_out_channel=32, filter_size=4, stride=2, name='conv2')
        fc1, fc1W = fc_layer(conv2, n_out=256, name='fc1')
        relu1 = tf.nn.relu(fc1)
        relu1_d = tf.nn.dropout(relu1, 0.5)

        # Add the action tensor in the 2nd hidden layer
        t2 = tflearn.fully_connected(action, 256)
        net = tflearn.activation(relu1_d + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.0003, maxval=0.0003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(agent): 
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value", episode_ave_max_q)

    vardict = {v.name: v for v in agent.actor.network_params if 'W' in v.name}
    for k,v in vardict.items():
        tf.scalar_summary(k, tf.sqrt(tf.nn.l2_loss(v)))

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars

class Agent:
    def __init__(self):
        self.debug = False
        self.sess = tf.Session()
        self.last_action, self.last_state = None, None

        self.EXPLORE_STEPS = 10000
        self.explore_flag = True
        self.epsilon = 1.0

        action_dim = 3
        self.actor = ActorNetwork(self.sess, action_dim, \
            ACTOR_LEARNING_RATE, TAU)
        self.critic = CriticNetwork(self.sess, action_dim, \
            CRITIC_LEARNING_RATE, TAU, self.actor.get_num_trainable_vars())

        self.sess.run(tf.initialize_all_variables())

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        # tried_but_not_worked: fill the replay memory (10000 frames) with human play data at the beginning of training
        # load_human_replay_mem(self.replay_buffer)

        # bookkeeping
        self.ep_ave_max_q = 0

    def act(self, state, last_reward, done, epCnt, human_action=None):
        # tried_but_not_worked: control the magnitude of the reward
        if done:
            assert last_reward == -100
            last_reward = -1
        # if human_action is not None, the agent learns human's behaviour 
        # by putting state transitions into replay memory
        learn_from_human = human_action is not None
        if learn_from_human:
            a = human_action
        else:
            a = self.actor.predict(np.expand_dims(state,0)).reshape(-1)
            # tried_but_not_worked add exploration with Ornstein-Uhlenbeck process as suggested by the DDPG (seems unhelpful)
            self.epsilon -= 1.0 / self.EXPLORE_STEPS
            noise_t = np.zeros([3])
            noise_t[0] = self.explore_flag * max(self.epsilon, 0) * OU(a[0],  0.0 , 0.60, 0.30)
            noise_t[1] = self.explore_flag * max(self.epsilon, 0) * OU(a[1],  0.5 , 1.00, 0.10)
            noise_t[2] = self.explore_flag * max(self.epsilon, 0) * OU(a[2], -0.1 , 1.00, 0.05)
            a += noise_t

        ep_ave_max_q_to_return = 0

        if self.last_state is None: # just use current state-action to initialize, although wrong
            self.last_state = state
            self.last_action = a

        # add only 10% experience into replay buffer
        if learn_from_human or np.random.rand()<0.1:
            self.replay_buffer.add(self.last_state.copy(), self.last_action.copy(), last_reward, done, state)

        # train the network on only 10% time steps
        if np.random.rand()<0.1 and self.replay_buffer.size() > MINIBATCH_SIZE:
            print ('Train network. replay mem size: %d' % self.replay_buffer.count)     
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replay_buffer.sample_batch(MINIBATCH_SIZE)

            # Calculate targets
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in xrange(MINIBATCH_SIZE):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + GAMMA * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
            
            self.ep_ave_max_q += np.amax(predicted_q_value)
            ep_ave_max_q_to_return = self.ep_ave_max_q
            if done:
                self.ep_ave_max_q = 0

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)                
            grads = self.critic.action_gradients(s_batch, a_outs)
            print(np.max(grads),np.min(grads))
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

            if self.debug == True:
                self.debug = not self.debug
                embed()

        self.last_state = state
        self.last_action = a
        return a, ep_ave_max_q_to_return

def OU(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)
def var_by_name(name):
    return [v for v in tf.all_variables() if v.name==name][0]
def restore(sess, chkpt_fname):
    vardict = {
    'conv1/W:0': 'po/conv1/W',
    'conv1/b:0': 'po/conv1/b',
    'conv1_1/W:0': 'po/conv1/W',
    'conv1_1/b:0': 'po/conv1/b',
    'conv1_2/W:0': 'po/conv1/W',
    'conv1_2/b:0': 'po/conv1/b',
    'conv1_3/W:0': 'po/conv1/W',
    'conv1_3/b:0': 'po/conv1/b',
    'conv2/W:0': 'po/conv2/W',
    'conv2/b:0': 'po/conv2/b',
    'conv2_1/W:0': 'po/conv2/W',
    'conv2_1/b:0': 'po/conv2/b',
    'conv2_2/W:0': 'po/conv2/W',
    'conv2_2/b:0': 'po/conv2/b',
    'conv2_3/W:0': 'po/conv2/W',
    'conv2_3/b:0': 'po/conv2/b',
    'fc1/W:0': 'po/fc1/W',
    'fc1/b:0': 'po/fc1/b',
    'fc1_1/W:0': 'po/fc1/W',
    'fc1_1/b:0': 'po/fc1/b',
    'fc1_2/W:0': 'po/fc1/W',
    'fc1_2/b:0': 'po/fc1/b',
    'fc1_3/W:0': 'po/fc1/W',
    'fc1_3/b:0': 'po/fc1/b',
    }
    for local, remote in sorted(vardict.items()):
        print 'restoring: %s  <-  %s' % (local, remote)
        saver=tf.train.Saver({remote:var_by_name(local)})
        saver.restore(sess, chkpt_fname)

def load_human_replay_mem(replay_buffer):
    print 'Loading human replay memory...'
    datadict=np.load('human_replay_mem.npy').item()
    replay_buffer.buffer=datadict['deque']
    replay_buffer.count = len(datadict['deque'])


def preprocess_state(rgb, flatten=False):
    grey =  np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grey = grey[:80,...] # crop the bottom (which are some game statistics, useless)
    grey -= 159.0 # substract mean (computed from my plays)
    return grey

if __name__ == '__main__':
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    human_play = False
    def key_press(k, mod):
        global human_play
        if k==key.H:     human_play  = not human_play
        if k==key.D:     agent.debug = True
        if k==key.A:     agent.actor.printLogits = not agent.actor.printLogits
        if k==key.E:     
            agent.explore_flag = not agent.explore_flag
            print 'explore_flag: %s, current epsilon: %f' % (agent.explore_flag, agent.epsilon)
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = my_car_env.CarRacing()
    env.render()
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    # ===========================
    #   Agent Training
    # ===========================
    envHelper = lib.EnvHelper()
    agent = Agent()
    summary_ops, summary_vars = build_summaries(agent)
    writer = tf.train.SummaryWriter(SUMMARY_DIR, agent.sess.graph)
    # tried_but_not_worked: initialize weights with my supervised-learning pretrained model at the beginning of training
    restore(agent.sess,'chkpts/reg0.005.ckpt')
    ep = 0
    while True:
        env.reset()
        ep_reward, steps, ep = 0, 0, ep + 1

        while True:
            if steps < 30:
                env.contactListener_keepref.lastTouchRoadTime = time.time() # so that the agent won't be killed if steps<30
                unprocessed_s, r, done, info = env.step(action=[0., 1., 0.])
            else:
                if human_play:
                    act_to_execute = a.copy()
                    _, ep_ave_max_q = agent.act(envHelper.get_state(), r, done, ep, act_to_execute)
                else:
                    act_to_execute, ep_ave_max_q = agent.act(envHelper.get_state(), r, done,ep)
                    if done:
                        break
                unprocessed_s, r, done, info = env.step(act_to_execute)
                env.render()

                # time for some bookkeeping
                if done:
                    summary_str = agent.sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(steps)
                    })
                    writer.add_summary(summary_str, ep)
                    print '| Reward: %.2i' % int(ep_reward), " | Episode", ep, \
                        '| Qmax: %.4f' % (ep_ave_max_q / float(steps))

            s = preprocess_state(unprocessed_s)
            envHelper.add_frame_to_state(s)
            ep_reward += r
            steps += 1
