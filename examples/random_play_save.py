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

NUM_EPISODES = 200

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

env = Environment(args.game)
gui = GUI(env.game_name(), env.n_channels, visualize=False)

e = 0
returns = []
num_actions = env.num_actions()

# Run NUM_EPISODES episodes and log all returns
while e < NUM_EPISODES:
    # Initialize the return for every episode
    G = 0.0

    # For generating the dataset
    t = 0
    ep_folder = Path('./generated_dataset/ep_%06d/' % e)
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

        plt.close('all')
        numerical_state = np.amax(s_prime * np.reshape(np.arange(gui.n_channels) + 1, (1, 1, -1)), 2) + 0.5
        image = gui.cmap(gui.norm(numerical_state))
        image = image[:, :, :3] # Simply drop the alpha channel
        image = (image * 255).round().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_folder / ('obs_%06d.png'%(t))), image)

        # print(image)
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()
        
        # plt.imshow(numerical_state, cmap=gui.cmap, norm=gui.norm, interpolation='none')
        # plt.savefig(str(ep_folder / ('obs_%06d.png'%(t))))

        G += reward
        t += 1

    # Increment the episodes
    e += 1

    # Store the return for each episode
    returns.append(G)

print("Avg Return: " + str(numpy.mean(returns))+"+/-"+str(numpy.std(returns)/numpy.sqrt(NUM_EPISODES)))


