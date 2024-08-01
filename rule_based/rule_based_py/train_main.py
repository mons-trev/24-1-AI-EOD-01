# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import copy
from scipy.special import softmax
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""# CONFIG"""

MEM_SIZE = 50000
MEM_SIZE_MIN = 1000

BATCH_SIZE = 64
GAMMA = 0.1 #gamma


LEARN_MAX = 0.005
LEARN_MIN = 0.00005
LEARN_EPOCH = 2500
LEARN_DECAY = 0.99


EPSILON = 0.9997
EPSILON_DECAY = 0.9997
EPSILON_MIN = 0.001

CONV_UNITS = 64
DENSE_UNITS = 512
UPDATE_TARGET_EVERY = 5

EPISODES = 60000
PRINT_EVERY=100
SAVE_EVERY=1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# Train"""

grid_size = (9, 9)
num_mines = 10

env = Environment(grid_size, num_mines)

agent = DQN_Agent(grid_size,  env.points, num_mines)

clears=0
rewards = []


rewards_list = []
clear_list = []
cnt_list = []
loss_list = []

ave_rewards_list = []
ave_clear_list = []
ave_count_list = []
ave_loss_list = []

mid_rewards_list = []
mid_clear_list = []
mid_cnt_list = []
mid_loss_list = []

lr_list = []

for episode in range(EPISODES):
    env.reset()
    state = env.present_state.copy()
    done = False
    clear = False
    total_reward = 0
    cnt = 0
    loss = 0
    agent.action=env.points.copy()
    loss_cnt=0
    loss_arr=[]
  
    while not done:
        action = agent.get_action(state)
        cnt+=1
        next_state, reward, done, clear = env.step(action)
        agent.append_sample(state, action, reward, next_state, done, clear)
        state = next_state.copy()

        total_reward += reward

        if len(agent.memory) > MEM_SIZE_MIN:
            loss = agent.train_model()
            loss = loss.item()
            loss_arr.append(loss)
            loss_cnt+=1
        if done or clear:
            if clear:
                clears+=1
            break

    rewards.append(total_reward)

    # 평가지표
    rewards_list.append(total_reward)
    ave_rewards_list.append(np.mean(rewards_list[-PRINT_EVERY*10:]))
    mid_rewards_list.append(np.median(rewards_list[-PRINT_EVERY*10:]))

    clear_list.append(clear)
    ave_clear_list.append(np.mean(clear_list[-PRINT_EVERY*10:]))
    mid_clear_list.append(np.median(clear_list[-PRINT_EVERY*10:]))

    cnt_list.append(cnt)
    ave_count_list.append(np.mean(cnt_list[-PRINT_EVERY*10:]))
    mid_cnt_list.append(np.median(cnt_list[-PRINT_EVERY*10:]))

    loss_list.append(loss)
    ave_loss_list.append(np.mean(loss_list[-PRINT_EVERY*10:]))
    mid_loss_list.append(np.median(loss_list[-PRINT_EVERY*10:]))

    lr_list.append(agent.optimizer.param_groups[0]['lr'])
    # lr 조절
    if (episode+1) % LEARN_EPOCH == 0:
        lr = agent.optimizer.param_groups[0]['lr'] * LEARN_DECAY
        agent.optimizer.param_groups[0]['lr'] = max(lr, LEARN_MIN)
    if ((episode+1) % SAVE_EVERY == 0) or episode+1 == EPISODES:

        # 시각화 저장
        df = pd.DataFrame()
        df['rewards'] = rewards_list
        df['ave_rewards'] = ave_rewards_list
        df['mid_rewards'] = mid_rewards_list
        df['clear'] = clear_list
        df['ave_clear'] = ave_clear_list
        df['mid_clear'] = mid_clear_list
        df['cnt'] = cnt_list
        df['ave_count'] = ave_count_list
        df['mid_count'] = mid_cnt_list
        df['loss'] = loss_list
        df['ave_loss'] = ave_loss_list
        df['mid_loss'] = mid_loss_list
        df['lr'] = lr_list

        with open(f'/content/gdrive/MyDrive/지뢰찾기/experiment1_visualizing_mine_{num_mines}.pkl', 'wb') as f:
            pickle.dump(df, f)


    if (episode+1) % PRINT_EVERY == 0:
        print(f"[{episode+1}/{EPISODES}]", end=" | ")
        print(f"avg clear: {round(np.mean(clear_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"cnt: {round(np.mean(cnt_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(cnt_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"Reward: {round(np.mean(rewards_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(rewards_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"loss: {round(np.mean(loss_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(loss_list[-PRINT_EVERY*10:]),3)}", end=' | ')
        print(f"Epsilon: {round(agent.epsilon, 5)}", end=" | ")
        print(f"lr: {round(agent.optimizer.param_groups[0]['lr'], 5)}")

print("Training completed.")

torch.save(agent.model.state_dict(), f"/content/gdrive/MyDrive/지뢰찾기/9by9mine3{num_mines}_{EPISODES}_changelearnrate_model_reward")
torch.save(agent.target_model.state_dict(), f"/content/gdrive/MyDrive/지뢰찾기/9by9mine3{num_mines}_{EPISODES}_changelearnrate_target_model_reward_memory")
torch.save(agent.memory, f"/content/gdrive/MyDrive/지뢰찾기/9by9mine3{num_mines}_{EPISODES}_changelearnrate_model_reward_memory")


"""# Train 시각화"""

# 시각화 코드

def visualizing(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(3, 4, figsize=(20, 15), squeeze=False)
    axs[0, 0].plot(df['ave_rewards'], color = 'blue')
    axs[0, 0].set_title("Avg Reward")
    axs[0, 1].plot(df['mid_rewards'], color = 'blue')
    axs[0, 1].set_title("Median Reward")
    axs[0, 2].plot(df['rewards'], color = 'blue')
    axs[0, 2].set_title("Reward")

    axs[1, 0].plot(df['ave_count'], color = 'skyblue')
    axs[1, 0].set_title("Avg Cnt")
    axs[1, 1].plot(df['mid_count'], color = 'skyblue')
    axs[1, 1].set_title("Median Cnt")
    axs[1, 2].plot(df['cnt'], color = 'skyblue')
    axs[1, 2].set_title("Cnt")

    axs[2, 0].plot(df['ave_loss'], color = 'green')
    axs[2, 0].set_title("Avg Loss")
    axs[2, 1].plot(df['mid_loss'], color = 'green')
    axs[2, 1].set_title("Median Loss")
    axs[2, 2].plot(df['loss'], color = 'green')
    axs[2, 2].set_title('Loss')

    axs[0, 3].plot(df['ave_clear'], color = 'red')
    axs[0, 3].set_title("Avg Clear")
    axs[1, 3].plot(df['lr'], color = 'grey')
    axs[1, 3].set_title("Learning Rate")

    plt.show()

visualizing(f'/content/gdrive/MyDrive/지뢰찾기/experiment1_visualizing_mine_{num_mines}.pkl')