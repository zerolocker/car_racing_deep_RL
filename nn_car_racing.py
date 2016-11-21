import sys, math, time
import numpy as np

from IPython import embed

import lib, my_car_env

render = False if len(sys.argv)>1 and sys.argv[1]=='norender' else True

def preprocess_state(rgb, flatten=False):
    grey =  np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grey = grey[:80,...] # crop the bottom (which are some game statistics, useless)
    grey -= 159.0 # substract mean (computed from my plays)
    return grey

if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        if k==key.D:     agent.debug=True
        if k==key.N:     agent.NN.debug=True
        if k==key.A:     agent.NN.printAct = not agent.NN.printAct
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
    
    # Initialize my agent's components
    envHelper = lib.EnvHelper()
    agent = lib.Agent()

    SKIP_FRAME = 5
    ep = 0
    while True:
        env.reset()
        total_reward, steps, ep = 0.0,0,ep+1

        while True:
            if steps < 30:
                env.contactListener_keepref.lastTouchRoadTime = time.time() # so that the agent won't be killed if steps<30
                unprocessed_s, r, done, info = env.step(action=[0., 1., 0.])
            else:
                a = agent.act(envHelper.get_state(), r, done); 
                a[1]=0.0; a[2]=0.0; # ignore agent's gas, brake action
                if done: break
                r = 0;
                for i in xrange(SKIP_FRAME):
                    unprocessed_s, r_f, done, info = env.step(a)
                    r += r_f
                    if render and not record_video: env.render() 
                    if done: break
            s = preprocess_state(unprocessed_s)
            envHelper.add_frame_to_state(s)

            total_reward += r
            if steps % 200 == 0 or done:
                pass
                # plt.imshow(s, cmap='gray')
                # plt.show()
                # plt.pause(0.01) # make the figure draw non-blockingly. using this you don't even need to call plt.ion() at the start
            if done:
                 print("episode {} (undiscounted)total_reward {:+0.2f}".format(ep, total_reward))
            steps += 1
    env.monitor.close()

