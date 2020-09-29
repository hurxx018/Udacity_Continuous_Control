import os

from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

import torch

from ddpg_agent import Agent


def train_ddpg_v1(
    env,
    agent,
    n_episodes = 2000
    ):

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    scores_deque = deque(maxlen = 100)
    scores = []
    # max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode = True)[brain_name] # reset the environment
        states = env_info.vector_observations               # get the current state (for each agent)
        num_agents = len(env_info.agents)
        # scores_agent = np.zeros(num_agents)                          # initialize the score (for each agent)
        scores_agent = 0.0
        agent.reset()
        while True:

            actions = agent.act(states, add_noise = True)

            env_info = env.step(actions)[brain_name] # send all actions to the environment

            next_states = env_info.vector_observations
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores_agent += rewards[0]
            if np.any(dones): # exit loop if episode finished
                break

        score = scores_agent #np.mean(scores_agent)
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(i_episode, np.mean(scores_deque)))
    return scores


def main():

    # load version 1 (with 1 agent) of the environment
    env_name = "Reacher_Windows_x86_64_version1\Reacher.exe" # Add the Unity Reacher Environment name
    no_graphics = True
    env = UnityEnvironment(file_name = env_name, no_graphics = no_graphics)

    # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the enviroment
    env_info = env.reset(train_mode = True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print("Number of agents : ", num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print("Size of each action : ", action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print("There are {} agents. Each observes a state with length: {}".format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random_seed = 12345 #10
    agent = Agent(state_size, action_size, random_seed, device = device)

    scores = train_ddpg_v1(env, agent, n_episodes = 300)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()
