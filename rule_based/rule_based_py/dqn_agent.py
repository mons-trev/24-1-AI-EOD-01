"""# Agent"""

class DQN_Agent:
    def __init__(self, state_size, action, num_mine, model=None):
        self.num_mine = num_mine
        self.state_size = state_size
        self.memory = deque(maxlen=MEM_SIZE)
        self.action = torch.tensor(action)
        self.epsilon = EPSILON
        self.n = state_size[0]


        # model, target model gpu 올리고 초기화
        self.model = DQN_Net(state_size, len(action)).to(device)
        self.target_model = DQN_Net(state_size, len(action)).to(device)
        self.update_target_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARN_MAX)


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def append_sample(self, state, action_param, reward, next_state, done, clear):
        self.memory.append((state, action_param[0] * self.n + action_param[1], reward, next_state, done, clear))

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        state = torch.tensor(state)
        if np.random.rand() <= self.epsilon:  # 무작위 탐색
            valid_indices = (state == -2).nonzero(as_tuple=True)

            rand_idx = np.random.randint(len(valid_indices[0]))
            x, y = valid_indices[0][rand_idx].item(), valid_indices[1][rand_idx].item()

            return x, y

        else:
            state = state.unsqueeze(0).unsqueeze(0).to(dtype=torch.float).to(device)

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
    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):

        if len(self.memory) < MEM_SIZE_MIN:
            return

        self.batch_size = min(BATCH_SIZE, len(self.memory))
        mini_batch = random.sample(self.memory, self.batch_size)


        states = torch.tensor([sample[0] for sample in mini_batch], dtype=torch.float).to(device)
        actions = torch.tensor([sample[1] for sample in mini_batch], dtype=torch.long).to(device)
        rewards = torch.tensor([sample[2] for sample in mini_batch], dtype=torch.float).to(device)
        next_states = torch.tensor([sample[3] for sample in mini_batch], dtype=torch.float).to(device)
        dones = torch.tensor([sample[4] for sample in mini_batch], dtype=torch.float).to(device)
        clears = torch.tensor([sample[5] for sample in mini_batch], dtype=torch.bool).to(device)

        states = states.unsqueeze(1).to(device)
        next_states = next_states.unsqueeze(1).to(device)

        self.model.train()
        self.target_model.eval()

        predicts = self.model(states) # (64, 81)
        one_hot_action= F.one_hot(actions, self.n * self.n) # [64,81]
        predicts = torch.sum(one_hot_action*predicts, axis=1) # [64] 각 데이터의 action 에 해당하는 q value


        with torch.no_grad():
            next_q_values = self.target_model(next_states).flatten(1)
            flat_next_states = next_states.squeeze(1).flatten(1)
            valid_mask = (flat_next_states == -2).float()

            valid_next_q_values = next_q_values * valid_mask
            valid_next_q_values[valid_mask == 0] = float('-inf')
            max_valid_q_values, _ = torch.max(valid_next_q_values, dim=-1)
            target_q_values = rewards + (torch.ones_like(dones) - dones) * GAMMA * max_valid_q_values

        loss = F.mse_loss (predicts, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # 엡실론 감소
        self.epsilon = self.epsilon * EPSILON_DECAY
        self.epsilon = max(self.epsilon, EPSILON_MIN)

        return loss

