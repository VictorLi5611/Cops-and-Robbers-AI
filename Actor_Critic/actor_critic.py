import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from GridWorld import GridWorld


class RbfFeaturizer():
    '''
        This class converts the raw state/obvervation features into
        RBF features. It does a z-score normalization and computes the
        Gaussian kernel values from randomly selected centers.
    '''

    def __init__(self, env, n_features=100):
        centers = np.array([convertStateToArray(env.reset()[0])
                            for _ in range(n_features)])
        self._mean = np.mean(centers, axis=0, keepdims=True)
        self._std = np.std(centers, axis=0, keepdims=True)
        self._centers = (centers - self._mean) / self._std
        self.n_features = n_features

    def featurize(self, state):
        z = state[None, :] - self._mean
        z = z / self._std
        dist = cdist(z, self._centers)
        return np.exp(- (dist) ** 2).flatten()

def convertStateToArray(s):
    # number of cops in vision, x_diff to bank, y_diff to bank
    return np.concatenate(([np.count_nonzero(s['vision'] == -1000.)], s['bank_pos'] - s['robber_pos']))

def evaluate(env, featurizer, W, policy_func, n_runs=10):
    '''
        Evaluate the policy given the parameters W and policy function.
        Run the environment several times and collect the return.
    '''
    all_returns = np.zeros([n_runs])
    for i in range(n_runs):
        observation, info = env.reset()
        return_to_go = 0
        while True:
            # Agent
            observation = featurizer.featurize(convertStateToArray(observation))
            action = policy_func(observation, W)

            observation, reward, terminated, truncated, info = env.step(action)
            return_to_go += reward
            if terminated or truncated:
                break
        all_returns[i] = return_to_go
    return np.mean(all_returns)

def softmaxPolicy(x, Theta):
    return np.random.choice(4, p=sp.special.softmax(np.transpose(Theta) @ x))

def logSoftmaxPolicyGradient(x, a, Theta):
    probabilities = sp.special.softmax(np.transpose(Theta) @ x)
    matrixA = np.zeros([len(Theta), len(Theta[0])])
    matrixB = np.zeros([len(Theta), len(Theta[0])])
    
    matrixA[:, a] = x
    # for JAX use the following
    # matrixA = matrixA.at[:, a].set(x)

    for i in range(len(matrixB[0])):
        matrixB[:, i] = x
        # for JAX use the following
        # matrixB = matrixB.at[:, i].set(x)

    gradient = matrixA - (probabilities * matrixB)
    return gradient

def ActorCritic(env, featurizer, eval_func, gamma=0.99, actor_step_size=0.005, critic_step_size=0.005, max_episodes=2000, evaluate_every=20):
    Theta = np.random.rand(featurizer.n_features, env.action_space.n)
    w = np.random.rand(featurizer.n_features)
    eval_returns = []

    for i in range(1, max_episodes + 1):
        s, _ = env.reset()
        terminated = truncated = False
        actor_discount = 1
        
        while not (terminated or truncated):
            # featurize the state
            s = convertStateToArray(s)
            s = featurizer.featurize(s)

            # sample action using the softmax policy defined in 3b
            action = softmaxPolicy(s, Theta)

            # take the action
            obs, reward, terminated, truncated, _ = env.step(action)

            # compute TD error
            td_error = reward + gamma * np.dot(featurizer.featurize(convertStateToArray(obs)), w) - np.dot(s, w)
            
            # update critic parameters
            w = np.add(w, critic_step_size * td_error * s)

            # update actor parameters using logSoftmaxPolicy1dGradient
            Theta = np.add(Theta, actor_step_size * td_error * actor_discount * logSoftmaxPolicyGradient(s, action, Theta)) 

            # set next state equal to the observation
            s = obs
            actor_discount *= gamma

        if i % evaluate_every == 0:
            eval_return = eval_func(env, featurizer, Theta, softmaxPolicy)
            eval_returns.append(eval_return)
    
    return Theta, w, eval_returns

def render_env(env, featurizer, W, policy_func):
    observation, info = env.reset()
    while True:
        env.render()
        observation = featurizer.featurize(convertStateToArray(observation))
        action = policy_func(observation, W)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return

def runACExperiments(featurizer, eval_func):
    def repeatExperiments(num_cops, vision, actor_step_size, critic_step_size):
        env = GridWorld(render_mode=None, num_cops=num_cops, vision=vision)
        eval_returns_step_sizes = np.zeros([n_runs, n_eval])
        
        for r in range(n_runs):
            Theta, w, eval_returns = ActorCritic(env, featurizer, eval_func, actor_step_size=actor_step_size, critic_step_size=critic_step_size, max_episodes=max_episodes, evaluate_every=evaluate_every)
            eval_returns_step_sizes[r] = eval_returns
    
        return np.mean(eval_returns_step_sizes, axis=0)

    n_runs = 10
    max_episodes = 12000
    evaluate_every = 25
    n_eval = max_episodes // evaluate_every # num of evaluation during training
    num_cops_list = [5,8,10]
    actor_step_size = 0.005
    critic_step_size = 0.005
    results = np.zeros([len(num_cops_list)*2, n_eval])

    for i in range(len(num_cops_list)):
        results[i] = repeatExperiments(num_cops_list[i], 4, actor_step_size, critic_step_size)

    for i in range(len(num_cops_list)):
        results[i + len(num_cops_list)] = repeatExperiments(num_cops_list[i], 2, actor_step_size, critic_step_size)

    plt.figure()
    for i in range(len(num_cops_list)):
        plt.plot(np.arange(n_eval) * evaluate_every, results[i], label = 'No. of Cops = {}'.format(num_cops_list[i]))
    plt.legend()
    plt.title('Average Rewards for Vision Radius 4')
    plt.xlabel('No. of Episodes')
    plt.ylabel('Evaluated Returns')
    plt.savefig('AC_plot_v4.png')

    plt.figure()
    for i in range(len(num_cops_list)):
        plt.plot(np.arange(n_eval) * evaluate_every, results[i + len(num_cops_list)], label = 'No. of Cops = {}'.format(num_cops_list[i]))
    plt.legend()
    plt.title('Average Rewards for Vision Radius 2')
    plt.xlabel('No. of Episodes')
    plt.ylabel('Evaluated Returns')
    plt.savefig('AC_plot_v2.png')
    return results