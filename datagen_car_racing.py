import sys, math, time, os
import numpy as np

from IPython import embed

import lib

render = False if len(sys.argv)>1 and sys.argv[1]=='norender' else True

def preprocess_state(rgb, flatten=False):
    grey =  np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grey = grey[:80,...] # crop the bottom (which are some game statistics, useless)
    grey -= 159.0 # substract mean (computed from my plays)
    return grey


class HumanPlayRecorder:
    def __init__(self, file_prefix=''):
        if not os.path.exists('humanplaydata'): os.mkdir('humanplaydata')
        self.fname = 'humanplaydata/'+ file_prefix + '_%s' % time.strftime('%H%M%m%d')
        self.dumpcount = 0
        self.bufsize = 1000
        self.statebuf = []
        self.actionbuf = []

    def recordHumanPlay(self, state, action):
        self.statebuf.append(state)
        self.actionbuf.append(action)
        if len(self.statebuf) >= self.bufsize:
            print 'dumped'
            np.save(self.fname + '_%03d.npy' % self.dumpcount,
                    {'state':np.array(self.statebuf,dtype=np.float16), 'action': np.array(self.actionbuf,dtype=np.float16)})
            self.statebuf = []
            self.actionbuf = []
            self.dumpcount += 1

    def readHumanPlay(self, prefix):
        files = filter(lambda s:s.startswith(prefix), os.listdir('humanplaydata'))
        files = sorted(files)
        assert len(files)>0, 'Cannot find any matches'
        states, actions = [], []
        for f in files:
            datadict=np.load('humanplaydata/'+f).item()
            states.append(datadict['state'])
            actions.append(datadict['action'])
        states_all = np.concatenate(states)
        actions_all = np.concatenate(actions)
        return states_all, actions_all



if __name__=="__main__":
    import my_car_env
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
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
    rec = HumanPlayRecorder()

    SKIP_FRAME = 1
    ep = 0
    while True:
        env.reset()
        total_reward, steps, ep = 0.0,0,ep+1

        while True:
            if steps < 30:
                env.contactListener_keepref.lastTouchRoadTime = time.time() # so that the agent won't be killed if steps<30
                unprocessed_s, r, done, info = env.step(action=[0., 1., 0.])
            else:
                executed_action = a.copy()
                rec.recordHumanPlay(envHelper.get_state(), executed_action)
                if done: break
                r = 0;
                for i in xrange(SKIP_FRAME):
                    unprocessed_s, r_f, done, info = env.step(executed_action)
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
            #if done:
            #     print("episode {} (undiscounted)total_reward {:+0.2f}".format(ep, total_reward))
            steps += 1
    env.monitor.close()

