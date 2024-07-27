# 하이퍼 파라미터
MEM_SIZE = 50000
MEM_SIZE_MIN = 1000

LEARN_MAX = 0.001
LEARN_MIN = 0.0001
LEARN_EPOCH = 50000
LEARN_DECAY = 0.75

GAMMA = 0.1

EPSILON = 0.999
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.01

BATCH_SIZE = 64
CONV_UNITS = 64
UPDATE_TARGET_EVERY = 5

grid_size = (9, 9)
num_mines = 10

PRINT_EVERY = 100
SAVE_EVERY = 1000

EPISODES = 100000

#################################### train code
env = Environment(grid_size, num_mines)
agent = DQN_Agent(env)

# 평가 지표
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

        # count 제한: 81
        if cnt > env.ncol*env.nrow:
            done = True

        # 18개 이상 까진 경우만 리플레이 메모리에 저장
        save = env.check_18_up(next_state)
        if save:
            agent.append_sample(state, action, reward, next_state, done, clear)

        # 학습
        if len(agent.memory) > MEM_SIZE_MIN:
            loss = agent.train_model()
            loss = loss.item()

        if done or clear:
            break

    # 타깃 모델 업데이트
    if episode % UPDATE_TARGET_EVERY == 0:
        agent.update_target_model()

    # lr 조절
    if (episode+1) % LEARN_EPOCH == 0:
        lr = agent.optimizer.param_groups[0]['lr'] * LEARN_DECAY
        agent.optimizer.param_groups[0]['lr'] = max(lr, LEARN_MIN)

    # 평가 지표 저장
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

    if ((episode+1) % SAVE_EVERY == 0) or episode+1 == EPISODES:
        # 리플레이 메모리 저장
        with open('memory.pkl', 'wb') as f:
            pickle.dump(agent.memory, f)

        # 모델 저장
        with open('model.pkl', 'wb') as f:
            pickle.dump(agent.model.state_dict(), f)

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

        with open('visualizing.pkl', 'wb') as f:
            pickle.dump(df, f)


    if (episode+1) % PRINT_EVERY == 0:
        print(f"[{episode+1}/{EPISODES}]", end=" | ")
        print(f"ave clear: {round(np.mean(clear_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"cnt: {round(np.mean(cnt_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(cnt_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"Reward: {round(np.mean(rewards_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(rewards_list[-PRINT_EVERY*10:]), 3)}", end=" | ")
        print(f"loss: {round(np.mean(loss_list[-PRINT_EVERY*10:]), 3)}/{round(np.median(loss_list[-PRINT_EVERY*10:]),3)}", end=' | ')
        print(f"Epsilon: {round(agent.epsilon, 5)}", end=" | ")
        print(f"lr: {round(agent.optimizer.param_groups[0]['lr'], 5)}")

print("Training completed.")
