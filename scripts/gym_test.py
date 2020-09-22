# import gym

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()

# import gym_tetris
# env = gym_tetris.make('Tetris-v0')
# env.reset()
# env.render()
#
# done = True
#
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#
# env.close()