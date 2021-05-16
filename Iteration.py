import numpy as np

class Gird():
    def __init__(self, size, gamma):
        self.size = size
        self.gamma = gamma
        self.T = [0.7, 0.1, 0.1, 0.1]
        self.action_list = ['up', 'down', 'left', 'right']
        self.epsilon = 0.0001

    def CrashWall(self, i, j):
        if i >= 0 and i < self.size and j >= 0 and j < self.size:
            return False
        return True

    def Next_Postion(self, i, j, action):
        pos = []
        if action == 'up':
            pos = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]
        elif action == 'down':
            pos = [[i + 1, j], [i - 1, j], [i, j - 1], [i, j + 1]]
        elif action == 'left':
            pos = [[i, j - 1], [i, j + 1], [i - 1, j], [i + 1, j]]
        elif action == 'right':
            pos = [[i, j + 1], [i, j - 1], [i - 1, j], [i + 1, j]]
        return pos

    def Act(self, i, j, action, U):
        if (i, j) == (7, 8):
            return 10
        if (i, j) == (2, 7):
            return 3
        U_t = 0
        pos = self.Next_Postion(i, j, action)
        R = np.zeros(4)
        for k in range(4):
            if self.CrashWall(pos[k][0], pos[k][1]):
                pos[k] = [i, j]
                R[k] = -1
            if (i, j) == (4, 3):
                R[k] = -5
            elif (i, j) == (7, 3):
                R[k] = -10
            U_t += self.T[k] * R[k] + self.gamma * self.T[k] * U[pos[k][0], pos[k][1]]
        return U_t

    def Display_Action(self, U):
        action_martix = [['any' for i in range(self.size)]for j in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in [(7, 8), (2, 7)]:
                    continue
                tmp = []
                for k in range(4):
                    tmp.append(self.Act(i, j, self.action_list[k], U))
                action_martix[i][j] = self.action_list[tmp.index(max(tmp))]
        for i in range(self.size):
                print(action_martix[i])

    def Value_Iter(self):
        count = 0
        U_pre = np.zeros((self.size, self.size))
        U_t = np.zeros((self.size, self.size))
        while True:
            for i in range(self.size):
                for j in range(self.size):
                    tmp = []
                    for k in range(4):
                        tmp.append(self.Act(i, j, self.action_list[k], U_pre))
                    U_t[i, j] = max(tmp)
            count += 1
            delta = np.mean(U_t - U_pre)
            if abs(delta) < self.epsilon:
                print("值迭代次数:%d,"%count)
                break
            U_pre=U_t.copy()
        print(U_t)
        print("策略")
        self.Display_Action(U_t)

    def Gauss_Seidel_Iter(self):
        count = 0
        U_t = np.zeros((self.size, self.size))
        while True:
            U_last = U_t.copy()
            for i in range(self.size):
                for j in range(self.size):
                    tmp = []
                    for k in range(4):
                        tmp.append(self.Act(i, j, self.action_list[k], U_t))
                    U_t[i, j] = max(tmp)
            count += 1
            delta = np.mean(U_t - U_last)
            if(abs(delta) < self.epsilon):
                print("Gauss-Seidel迭代次数:%d,"%count)
                break
        print(U_t)
        print("策略")
        self.Display_Action(U_t)

    def compute_U_pi_k(self, policy, U_pre):
        
        U_t = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                tmp = 0
                if policy[i, j] == 0:
                    tmp = self.Act(i, j, 'up', U_pre)
                elif policy[i, j] == 1:
                    tmp = self.Act(i, j, 'down', U_pre)
                elif policy[i, j] == 2:
                    tmp = self.Act(i, j, 'left', U_pre)
                elif policy[i, j] == 3:
                    tmp = self.Act(i, j, 'right', U_pre)
                U_t[i, j] = tmp
        return U_t
    
    def Policy_Iter(self):
        _action_ = ['up', 'down', 'left', 'right']
        count = 0
        policy_pre = np.random.randint(4, size = (self.size, self.size))
        # U_pi = np.zeros((self.size, self.size))
        U_pi = self.compute_U_pi_k(policy_pre, np.zeros((self.size, self.size)))
        policy_t = policy_pre.copy()
        while True:
            flag = True
            U_pi = self.compute_U_pi_k(policy_pre, U_pi)
            for i in range(self.size):
                for j in range(self.size):
                    tmp = []
                    for k in range(4):
                        tmp.append(self.Act(i, j, self.action_list[k], U_pi))
                    policy_t[i, j] = tmp.index(max(tmp))
                    if (policy_t[i, j] != policy_pre[i, j]) and ((i, j) not in [(7, 8), (2, 7)]):
                        flag = False
            count += 1
            if flag:
                print("策略迭代次数:%d"%count)
                print(U_pi)
                break
            policy_pre = policy_t.copy()
        print("策略")
        for i in range(self.size):
            for j in range(self.size):
                if j == 9:
                    print(_action_[policy_t[i, j]])
                else:
                    print(_action_[policy_t[i, j]], end = ', ')

np.set_printoptions(precision=2,suppress=True)
gird = Gird(size = 10, gamma = 0.9)
gird.Value_Iter()
gird.Gauss_Seidel_Iter()
gird.Policy_Iter()
    
