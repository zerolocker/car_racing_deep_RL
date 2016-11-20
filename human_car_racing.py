import sys, math, time
import numpy as np

from IPython import embed

import lib, my_car_env
import matplotlib.pyplot as plt


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
        if k==key.D:  restart=True
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
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    
    running_mean = 0.0
    nFrame = 0
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            if steps<100:
                env.contactListener_keepref.lastTouchRoadTime = time.time() # so that the agent won't be killed if ststeps<100
            s, r, done, info = env.step(a)
            print(a)
            total_reward += r
            
            nFrame += 1
            running_mean += (np.mean(preprocess_state(s)) - running_mean) / nFrame
            if nFrame % 1000 == 0: print running_mean

            if steps % 200 == 0 or done:
                pass
                #plt.imshow(preprocess_state(s), cmap='gray')
                #plt.pause(0.01) # make the figure draw non-blockingly. using this you don't even need to call plt.ion() at the start
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                #print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.monitor.close()

