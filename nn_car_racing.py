import sys, math, time
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.envs.classic_control import rendering
from gym.utils import colorize, seeding

import pyglet
from pyglet.gl import *
import matplotlib.pyplot as plt
from IPython import embed

import lib


def preprocess_state(rgb, flatten=False):
    grey =  np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grey = grey[:80,...] # crop the bottom (which are some game statistics, useless)
    grey -= 153.0 # substract mean (computed from my plays)
    return grey

if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = gym.make('CarRacing-v0')
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    
    # Initialize my agent's components
    envHelper = lib.EnvHelper()
    agent = lib.Agent()

    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False

        while True:
            if steps < 100: 
                unprocessed_s, r, done, info = env.step(action=[0., 0., 0.])
            else:
                a = agent.act(envHelper.get_state(), r)
                unprocessed_s, r, done, info = env.step(a)
            s = preprocess_state(unprocessed_s)
            envHelper.add_frame_to_state(s)

            total_reward += r
            if steps == 105:
                import tensorflow as tf
                vlist = tf.all_variables()
                print sorted([np.linalg.norm(v) for v in agent.NN._sess.run(vlist)],reverse=True)
                print 'look at the norm, they are so big'
                time.sleep(10)
            if steps % 200 == 0 or done:
                # plt.imshow(s, cmap='gray')
                # plt.show()
                # plt.pause(0.01) # make the figure draw non-blockingly. using this you don't even need to call plt.ion() at the start
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.monitor.close()

