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


"""# Test 시각화"""

def test_visualizing(file):
    with open(file, 'rb') as f:
        df = pickle.load(f)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)
    axs[0, 0].plot(df['avg_rewards'], color = 'blue')
    axs[0, 0].set_title("Avg Reward")
    axs[0, 1].plot(df['mid_rewards'], color = 'blue')
    axs[0, 1].set_title("Median Reward")
    axs[0, 2].plot(df['rewards'], color = 'blue')
    axs[0, 2].set_title("Reward")

    axs[1, 0].plot(df['avg_count'], color = 'skyblue')
    axs[1, 0].set_title("Avg Cnt")
    axs[1, 1].plot(df['mid_count'], color = 'skyblue')
    axs[1, 1].set_title("Median Cnt")
    axs[1, 2].plot(df['cnt'], color = 'skyblue')
    axs[1, 2].set_title("Cnt")

    axs[0, 3].plot(df['avg_clear'], color = 'red')
    axs[0, 3].set_title("Avg Clear")

    plt.show()

class DQN_Agent:
    def __init__(self, state_size, action, num_mine, model=None):
        self.num_mine = num_mine
        self.state_size = state_size
        self.action = torch.tensor(action)
        self.n = state_size[0]


        self.model = DQN_Net(state_size, len(action)).to(device)
        self.model.load_state_dict(torch.load(f"/content/gdrive/MyDrive/지뢰찾기/9by9mine3{num_mines}_200000_changelearnrate_model_reward_final", map_location = device))

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
            state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(dtype=torch.float).to(device)


            with torch.no_grad():
                q_values = self.model(state).flatten()

                flat = state.squeeze(0).flatten()
                mask = (flat == -2).float()

                masked_q_values = q_values * mask
                masked_q_values[mask == 0] = float('-inf')

                max_q_value, action_idx = torch.max(masked_q_values, dim=-1)

                x = action_idx // self.n
                y = action_idx % self.n

                self.action = [item for item in self.action if item != (x, y)]
                return x, y

grid_size = (9, 9)
num_mines = 10

env = Environment(grid_size, num_mines)

agent = DQN_Agent(grid_size,  env.points, num_mines)

total_clears = 0
clears=0

rewards_list = []
clear_list = []
cnt_list = []

avg_rewards_list = []
avg_clear_list = []
avg_count_list = []

mid_rewards_list = []
mid_clear_list = []
mid_cnt_list = []

total_cnt = []
tcnt = 0

EPISODES = 10000

for episode in range(EPISODES):
    # reset
    env.reset()
    state = env.present_state.copy()
    done = False
    clear = False
    total_reward = 0
    cnt = 0
    loss = 0
    agent.action = env.points.copy()

    # 게임 종료까지 반복
    while not done:
        cnt+=1

        state = env.present_state.copy()
        action = agent.get_action(state)
        next_state, reward, done, clear = env.step(action)
        total_reward += reward

        if done or clear:
            if clear:
                tcnt+=1
            break

    # 평가지표
    rewards_list.append(total_reward)
    avg_rewards_list.append(np.mean(rewards_list[-PRINT_EVERY:]))
    mid_rewards_list.append(np.median(rewards_list[-PRINT_EVERY:]))

    clear_list.append(clear)
    avg_clear_list.append(np.mean(clear_list[-PRINT_EVERY:]))
    mid_clear_list.append(np.median(clear_list[-PRINT_EVERY:]))

    cnt_list.append(cnt)
    avg_count_list.append(np.mean(cnt_list[-PRINT_EVERY:]))
    mid_cnt_list.append(np.median(cnt_list[-PRINT_EVERY:]))

    if ((episode+1) % SAVE_EVERY == 0) or episode+1 == EPISODES:
        # 시각화 저장
        df = pd.DataFrame()
        df['rewards'] = rewards_list
        df['avg_rewards'] = avg_rewards_list
        df['mid_rewards'] = mid_rewards_list
        df['clear'] = clear_list
        df['avg_clear'] = avg_clear_list
        df['mid_clear'] = mid_clear_list
        df['cnt'] = cnt_list
        df['avg_count'] = avg_count_list
        df['mid_count'] = mid_cnt_list

        with open(f'/content/gdrive/MyDrive/test.pkl', 'wb') as f:
            pickle.dump(df, f)


    if (episode+1) % PRINT_EVERY == 0:
        total_cnt.append(tcnt)
        tcnt = 0
        print(f"[{episode+1}/{EPISODES}]", end=" | ")
        print(f"avg clear: {round(np.mean(clear_list[-PRINT_EVERY:]), 3)}", end=" | ")
        print(f"cnt: {round(np.mean(cnt_list[-PRINT_EVERY:]), 3)}/{round(np.median(cnt_list[-PRINT_EVERY:]), 3)}", end=" | ")
        print(f"Reward: {round(np.mean(rewards_list[-PRINT_EVERY:]), 3)}/{round(np.median(rewards_list[-PRINT_EVERY:]), 3)}", end="\n")
        env.render()

print(f"Test completed. avg win rate: {round(np.mean(clear_list), 3)}")

print(np.median(total_cnt))

test_visualizing(f'/content/gdrive/MyDrive/test.pkl')