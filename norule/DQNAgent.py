class DQN_Agent:
    def __init__(self, env):
        self.env = env

        # 환경 관련 파라미터
        self.num_mine = self.env.num_mine
        self.state_size = self.env.gridworld_size
        self.nrow = self.env. nrow
        self.ncol = self.env.ncol
        self.n = self.nrow*self.ncol

        # action, q_values
        self.action = torch.tensor(self.env.points)
        self.q_values = torch.zeros(self.action.shape, dtype=torch.float32)

        # 하이퍼파라미터
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        # 리플레이 메모리
        self.memory = deque(maxlen = MEM_SIZE)

        # device 지정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model, target model 초기화
        self.model = DQN_Net(state_size, len(action), CONV_UNITS).to(self.device)
        self.target_model = DQN_Net(state_size, len(action), CONV_UNITS).to(self.device)
        self.update_target_model()

        # 손실함수
        self.criterion = nn.MSELoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARN_MAX)


    def update_target_model(self):
        # 타깃 모델을 업데이트 하는 함수
        self.target_model.load_state_dict(self.model.state_dict())


    def append_sample(self, state, action, reward, next_state, done, clear):
        # 샘플을 리플레이 메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done, clear))


    def get_action(self, state):
        state=torch.tensor(state).to(self.device)

        if np.random.rand() <= self.epsilon:  # 무작위 탐색
            act = random.choice(self.action)

        else :
            state = state.unsqueeze(0).to(dtype = torch.float32)
            state = state.unsqueeze(0)
            # 정규화
            state = state/8

            with torch.no_grad():
                q_values = self.model(state).flatten().to("cpu")
                max_idx = torch.argmax(q_values)
                act = max_idx.item()

                self.q_values = q_values

        return act


    def train_model(self):
        #리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # 상태 정규화 포함
        states = torch.tensor([sample[0]/8 for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        actions = torch.tensor([sample[1] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        rewards = torch.tensor([sample[2] for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1)
        next_states = torch.tensor([sample[3]/8 for sample in mini_batch], dtype=torch.float32).to(self.device).reshape(-1,1,self.nrow,self.ncol)
        dones = torch.tensor([sample[4] for sample in mini_batch], dtype=torch.long).to(self.device).reshape(-1,1)
        clears = torch.tensor([sample[5] for sample in mini_batch], dtype=torch.bool).reshape(-1,1)

        # Q(s,a) 값을 예측값으로 사용 - (batch, action_space.n)
        pred_q_values = self.model(states).gather(1, actions) # action idx의 데이터만 꺼냄

        # target 값 계산 : reward + gamma * Q(s',a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1).values.reshape(-1,1)
            target_q_values = rewards + (torch.ones(next_q_values.shape, device=self.device) - dones) * self.gamma * next_q_values

        # 오류 함수를 줄이는 방향으로 모델 업데이트
        loss = F.mse_loss(pred_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward() # 역전파
        self.optimizer.step() # 계산한 기울기를 optimizer에 맞추어 가중치를 수정

        # epsilon update
        self.epsilon = self.epsilon*self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss
