"""# 환경"""

class Environment:
    def __init__(self, gridworld_size:Tuple, num_mine:int):

        self.gridworld_size = gridworld_size
        self.nrow, self.ncol = self.gridworld_size
        self.num_mine = num_mine
        self.totalcnt = 0

        # 그리드월드의 좌표(튜플)의 리스트
        self.points = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.points.append((i,j))

        self.num_actions = len(self.points)

        self.reward_dict = {'mine': -8, 'empty':1, 'clear':5}

        # 지뢰 랜덤으로 배정
        self.mine_points = random.sample(self.points, self.num_mine)

        # 그리드 월드 rendering (지뢰: 'M')
        self.gridworld = np.full(shape=(self.nrow, self.ncol), fill_value=".")
        for x,y in self.mine_points:
            self.gridworld[x,y] = 'M'

        # 지뢰 = True인 맵
        self.mine_bool = (self.gridworld=='M')

        # 주변 지뢰 개수를 표시한 맵 (지뢰 위치: -1)
        self.map_answer = np.zeros(self.gridworld_size)

        for x,y in self.points:
            cnt = self.check_mine((x,y))
            self.map_answer[x,y] = cnt


        # state 맵
        self.present_state = np.full((self.nrow, self.ncol), -2) # BFS로 탐색하지 않은 부분을 -2로 초기화


    def check_mine(self, coord:Tuple):

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        x, y = coord
        result = 0

        if self.mine_bool[x,y]:
            result = -1

        else:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
                    if self.mine_bool[nx, ny]:
                        result += 1
        return int(result)



    def bfs_minesweeper(self, clicked_point:Tuple):

        queue = deque([clicked_point])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        result = self.present_state.copy()
        switch_cnt=0

        while queue:
            x, y = queue.popleft()
            if result[x, y]!=-2:
                continue
            result[x, y] = self.map_answer[x, y]
            switch_cnt+=1
            if self.map_answer[x,y] == 0:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.nrow and 0 <= ny < self.ncol and not result[nx, ny]!=-2:
                        queue.append((nx, ny))
        self.totalcnt += 1

        return result, switch_cnt


    def gridworld_reset(self):

        self.gridworld = np.full(shape=(self.nrow, self.ncol), fill_value=".")
        for x,y in self.mine_points:
            self.gridworld[x,y] = 'M'

        # 지뢰 = True인 맵
        self.mine_bool = (self.gridworld=='M')

        # 주변 지뢰 개수를 표시한 맵 (지뢰 위치: -1)
        self.map_answer = np.zeros(self.gridworld_size)
        for x,y in self.points:
            cnt = self.check_mine((x,y))
            self.map_answer[x,y] = cnt

        self.present_state = np.full((self.nrow, self.ncol), -2)
        self.totalcnt = 0


    def move_mine(self, action:Tuple):

        empty_points = list(set(self.points) - set(self.mine_points))
        new_mine = random.sample(empty_points, 1)

        self.mine_points.remove(action)
        self.mine_points.append(new_mine[0])

        self.gridworld_reset()


    def step(self, action:Tuple):

        x, y = action
        flag=True

        # 첫번째 action인 경우
        if np.count_nonzero(self.present_state==-2) == self.nrow*self.ncol :
            if action in self.mine_points:
                # 만약 start 좌표에 지뢰가 있는 경우 옮기기
                flag=False
                self.move_mine(action)


        # action에 따라 계산된 state
        next_state, switch_cnt = self.bfs_minesweeper(action)

        # 현재 위치 업데이트, 경로 추가
        self.present_state = next_state.copy()

        # ======
        # reward

        clear = done = False
        if action in self.mine_points:
            # 지뢰 밟은 경우 -> 지뢰찾기 실패
            # 음수의 보상과 함께 에피소드 종료
            reward = self.reward_dict['mine']
            done = True

        elif np.count_nonzero(self.present_state==-2) == self.num_mine :
            reward = self.reward_dict['clear']
            clear = True # 성공했는지 여부 판단을 위해
            done = True

        else :
            reward = self.reward_dict['empty']

            pred = 0
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.nrow and 0 <= ny < self.ncol and self.present_state[nx, ny]!=-2:
                    pred+=1

            if pred > 0 :
                reward = self.reward_dict['empty'] * 3

            else:
                reward = self.reward_dict['empty']
            done = False


        return next_state, reward, done, clear


    def reset(self):
        # 지뢰 랜덤으로 배정
        self.mine_points = random.sample(self.points, self.num_mine)
        self.gridworld_reset()
        self.totalcnt = 0


    def render(self):
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for (i,j) in self.points:
            if self.present_state[i,j] == -2:
                continue
            elif self.present_state[i,j] == -1:
                render_state[i,j] = "M"
            else:
                render_state[i,j] = self.present_state[i,j]

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_answer(self):
        render_state = np.full(shape=(self.nrow, self.ncol), fill_value=".")

        for (i,j) in self.points:
            if self.map_answer[i,j] == -1:
                render_state[i,j] = "M"
            else:
                render_state[i,j] = str(self.map_answer[i,j])

        render_state = pd.DataFrame(render_state)
        render_state = render_state.style.applymap(self.render_color)
        display(render_state)


    def render_color(self, var):
        color = {'0':'black', '1':"skyblue", '2':'lightgreen', '3':'red', '4':'violet', '5':'brown',
                 '6':'turquoise', '7':'grey', '8':'black', 'M':'white', '.':'black'}
        return f"color: {color[var]}"

    def train_reset(self, samples:list):
        self.mine_points = random.sample(samples, 1)
        self.gridworld_reset()
