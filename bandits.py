import numpy as np

class KArmedBandit:

    def __init__(self, num_arms = 10):
        self.num_arms = num_arms
        self.mean_rewards = np.random.normal(loc = 0.0, scale = 1.0, size=(self.num_arms,))

    def pull(self, choice):
        if 0 <= choice < self.num_arms:
            return np.random.normal(loc = self.mean_rewards[choice], scale=1.0)
        else:
            return -np.inf

class SimpleBanditAlgorithm:

    def __init__(self, num_arms, epsilon = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.reset()

    def __call__(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.Q)

    def reset(self):
        self.cumulative_reward = 0.0
        self.Q = np.zeros(self.num_arms)
        self.N = np.zeros(self.num_arms)

    def update(self, action, reward):
        self.cumulative_reward += reward
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])    


class OptimisticBanditAlgorithm:

    def __init__(self, num_arms, init_val = 10.0, epsilon = 0.0):
        self.num_arms = num_arms
        self.init_val = init_val
        self.epsilon = epsilon
        self.reset()

    def __call__(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.Q)

    def reset(self):
        self.cumulative_reward = 0.0
        self.Q = np.zeros(self.num_arms) + self.init_val
        self.N = np.zeros(self.num_arms)

    def update(self, action, reward):
        self.cumulative_reward += reward
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

class UCBBanditAlgorithm:

    def __init__(self, num_arms, c, epsilon = 0.0):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.t = 0
        self.c = c
        self.reset()

    def __call__(self):
        self.t += 1
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            ucb_estimates = np.zeros_like(self.Q)
            for a in range(self.num_arms):
                if self.N[a] == 0:
                    return a
                else:
                    ucb_estimates[a] = self.Q[a] + self.c * np.sqrt(np.log(self.t)/self.N[a])
            
            return np.argmax(ucb_estimates)

    def reset(self):
        self.t = 0
        self.cumulative_reward = 0.0
        self.Q = np.zeros(self.num_arms)
        self.N = np.zeros(self.num_arms)

    def update(self, action, reward):
        self.cumulative_reward += reward
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])

class GradientBanditAlgorithm:

    def __init__(self, num_arms, alpha, epsilon = 0.0):
        self.num_arms = num_arms
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self):
        self.t += 1
        probs = np.exp(self.H) / np.sum(np.exp(self.H))
        return np.argmax(probs)

    def reset(self):
        self.cumulative_reward = 0.0
        self.H = np.zeros(self.num_arms)
        self.reward_average = 0.0
        self.t = 0

    def update(self, action, reward):
        probs = np.exp(self.H) / np.sum(np.exp(self.H))
        self.cumulative_reward += reward
        self.reward_average += (1.0 / self.t) * (reward - self.reward_average)
        for a in range(self.num_arms):
            if a == action:
                self.H[a] += self.alpha * (reward - self.reward_average) * (1.0 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - self.reward_average) * probs[a]
        
class BanditProblem:

    def __init__(self, algorithm, num_arms = 10, run_size = 1000, num_runs = 2000):
        self.problem = KArmedBandit(num_arms)

        self.algorithm = algorithm
        self.history = { k : np.zeros((num_runs, run_size)) for k in algorithm }
            
        self.run_size = run_size
        self.num_runs = num_runs

    def run(self):
        run_reward_history = { k : np.zeros(self.run_size) for k in self.algorithm }
        
        for i in range(self.run_size):
            for alg in self.algorithm:
                a = self.algorithm[alg]()
                reward = self.problem.pull(a)
                self.algorithm[alg].update(a, reward)
                run_reward_history[alg][i] = reward

        return run_reward_history

    def multirun(self):
        for i in range(self.num_runs):
            for alg in self.algorithm:
                self.algorithm[alg].reset()
            res = self.run()
            for alg in self.algorithm:
                self.history[alg][i] = res[alg]
    

    


        
