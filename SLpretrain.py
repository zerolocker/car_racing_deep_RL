import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from IPython import embed
from helper import printdebug
import ipdb, argparse, time, os
from datagen_car_racing import HumanPlayRecorder
import lib

# Some hyperparameter setups

BATCH_SIZE = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, dest='epochs', help='num epochs', default=600)
parser.add_argument('--checkpoint-dir', type=str, dest='checkpoint_dir', help='dir to save checkpoint in', default='chkpts/')
parser.add_argument('--model-prefix', type=str, dest='model_prefix', help='filename prefix of saved checkpoint', default='noprefix')
parser.add_argument('--restore-chkpt', type=str, dest='restore_chkpt', help='path to checkpoint to restore training"', default='')
options = parser.parse_args()

paramStr = '%s' % (options.model_prefix)
logfile = open('out/'+paramStr+'.log', 'w+')
if not os.path.exists(options.checkpoint_dir): os.mkdir(options.checkpoint_dir)



# read training data and shuffle

print 'Reading all training data...'
rec = HumanPlayRecorder()
states_unshuffled, actions_unshuffled = rec.readHumanPlay('')
print 'Completed. shape: %s' % str(states_unshuffled.shape)
assert states_unshuffled.shape[0] == actions_unshuffled.shape[0]
NUM_EXAMPLES = states_unshuffled.shape[0]

print 'Shuffling...'
p = np.random.permutation(NUM_EXAMPLES)
states =  states_unshuffled[p]
actions_realval = actions_unshuffled[p]
actions_discrete = np.apply_along_axis(lib.map_to_discrete, axis=1, arr=actions_realval).astype(np.int32)
print 'Completed.'



# Construct neural network

sess = tf.Session()
policynet = lib.NN(sess, None, lr = 0.001)
saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='po'))

if options.restore_chkpt != '':
    printdebug('Resuming checkpoint: ' + options.restore_chkpt, logfile)
    saver.restore(sess, options.restore_chkpt)



# Init some variables used in training

MAX_ITER = options.epochs * NUM_EXAMPLES / BATCH_SIZE
duration = 0
chkpt_fname = options.checkpoint_dir+'/'+paramStr

batch_ptr = 0
reward_batch = np.ones([BATCH_SIZE], dtype=np.float32)
fc1W = [v for v in tf.all_variables() if '/W:0' in v.name and 'fc1' in v.name][0] # used to monitor its norm
fc1Wnorm = tf.reduce_sum(tf.square(fc1W))                                         # to monitor overfitting

printdebug('Training starts! NUM_EXAMPLES: %d BATCH_SIZE: %d' % (NUM_EXAMPLES,BATCH_SIZE), logfile)
for it in xrange(1, MAX_ITER+1):
    epoch = (it * BATCH_SIZE) / float(NUM_EXAMPLES)

    # construct a batch
    if batch_ptr + BATCH_SIZE > NUM_EXAMPLES: 
        batch_ptr = 0
    state_batch = states[batch_ptr:batch_ptr+BATCH_SIZE, ...]
    action_batch = actions_discrete[batch_ptr:batch_ptr+BATCH_SIZE, ...]
    batch_ptr += BATCH_SIZE

    # run backprop op
    start_time = time.time()
    _, loss, reg_loss_np, fc1Wnorm_np = policynet._sess.run(
        [policynet.train_op, policynet.loss, policynet.reg_loss, fc1Wnorm], feed_dict={
            policynet.state:             state_batch,
            policynet.true_action_steer: action_batch[:, 0],
            policynet.true_action_gas:   action_batch[:, 1],
            policynet.true_action_break: action_batch[:, 2],
            policynet.reward:            reward_batch,
        })
    printdebug('NN loss: %f reg_loss: %f fc1Wnorm: %f' % (loss, reg_loss_np, fc1Wnorm_np), logfile)
    duration += time.time() - start_time

    if it % 10 == 0:
        printdebug('epoch: {:.3f}, {:d}/{:d}, elapsed: {:.1f}s'.format(epoch, it, MAX_ITER, duration), logfile)
        duration = 0
    if it % 1000 == 0:
        saver.save(sess, '%s_%d.ckpt' % (chkpt_fname, it))

saver.save(sess, '%s.ckpt' % (chkpt_fname))
