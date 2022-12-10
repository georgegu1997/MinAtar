################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 random_play.py -g <game>                                                                             #                                                              #
################################################################################################################
import os
from pathlib import Path
import random, numpy, argparse
import numpy as np
from minatar import Environment
from minatar.gui import GUI
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
parser.add_argument("--n_epochs", "-e", type=int, default=1000)
args = parser.parse_args()

env = Environment(args.game)
gui = GUI(env.game_name(), env.n_channels, visualize=False)

e = 0
returns = []
num_actions = env.num_actions()

# Run args.n_epochs episodes and log all returns
while e < args.n_epochs:
    # Initialize the return for every episode
    G = 0.0

    # For generating the dataset
    t = 0
    ep_folder = Path('./generated_dataset/%s/ep_%06d/' % (args.game, e))
    if not os.path.exists(ep_folder):
        os.makedirs(ep_folder)

    # Initialize the environment
    env.reset()
    terminated = False

    #Obtain first state, unused by random agent, but inluded for illustration
    s = env.state()
    while(not terminated):
        # Select an action uniformly at random
        action = random.randrange(num_actions)

        # Act according to the action and observe the transition and reward
        reward, terminated = env.act(action)

        # Obtain s_prime, unused by random agent, but inluded for illustration
        s_prime = env.state()

        # plt.close('all')
        # numerical_state = np.amax(s_prime * np.reshape(np.arange(gui.n_channels) + 1, (1, 1, -1)), 2) + 0.5
        # image = gui.cmap(gui.norm(numerical_state))
        # image = image[:, :, :3] # Simply drop the alpha channel
        # image = (image * 255).round().astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(str(ep_folder / ('obs_%06d.png'%(t))), image)

        # state = s_prime
        # numerical_state = np.amax(state * np.reshape(np.arange(gui.n_channels) + 1, (1, 1, -1)), 2) + 0.5
        # plt.imshow(numerical_state, cmap=gui.cmap, norm=gui.norm, interpolation='none')
        # plt.show()

        # seg = env.env.state_seg()
        # print(seg.shape)
        # print(seg.dtype)
        # print(seg.max())
        # print(seg.min())
        # fig, axes = plt.subplots(3, 4)
        # for i in range(seg.shape[2]):
        #     axes[i//4, i%4].imshow(seg[:, :, i], cmap='gray')
        # axes[-1, -1].imshow(numerical_state, cmap=gui.cmap, norm=gui.norm, interpolation='none')
        # plt.show()

        # plt.imshow(s_prime)
        # plt.show()
        
        # plt.imshow(numerical_state, cmap=gui.cmap, norm=gui.norm, interpolation='none')
        # plt.savefig(str(ep_folder / ('obs_%06d.png'%(t))))

        image = s_prime
        image = (image * 255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_folder / ('obs_%06d.png'%(t))), image)

        G += reward
        t += 1

    # Increment the episodes
    e += 1

    # Store the return for each episode
    returns.append(G)

print("Avg Return: " + str(numpy.mean(returns))+"+/-"+str(numpy.std(returns)/numpy.sqrt(args.n_epochs)))


