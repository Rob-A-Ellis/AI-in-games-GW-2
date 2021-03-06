################ Environment ################

import numpy as np
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

        
class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        # Calls super constructor for the FrozenLake environment
        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

        self.probs = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.rewards = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)

        # Calculates the probabilities and expected rewards
        self.calc_probs()
        self.calc_rewards()

        # Checks the calculated probabilities against the target
        self.check_probs()

    def calc_probs(self):
        # Steps through all of the possible states
        for state in range(self.n_states):
            # Checks if the agent is in an abosorbing or end state
            if state == self.absorbing_state or self.lake_flat[state] in ('#','$'):
                self.probs[self.absorbing_state,state,:] = 1
                continue
            for action in range(self.n_actions):
                # Calculates the probabilities and factors in the slip chance
                for slip_action in range(self.n_actions):
                    next_state = self.act(slip_action, state)
                    self.probs[next_state,state,action] += self.slip/self.n_actions
                    if action == slip_action:
                        self.probs[next_state,state,action] += 1 - self.slip

    def check_probs(self):
        # Loads the target probabilities
        target = np.load('p.npy')

        # Checks all of the calculated probs against all of the targets
        if self.probs.all() == target.all():
            print("Probs match target")
        else:
            print("Probs differ from target")

    def calc_rewards(self):
        # Steps through all of the possible states
        for state in range(self.n_states):
            if state != self.absorbing_state:
                # Sets expected reward to 1 only if the agent is at the goal state
                if self.lake_flat[state] == '$':
                    self.rewards[self.absorbing_state,state,:] = 1

    def act(self, action, state):
        _, self.columns = self.lake.shape

        # Checks if the action being taken is possible or not
        if state - self.columns < 0:
            if action == 0:
                return state
        if state % self.columns == 0:
            if action == 1:
                return state
        if state + self.columns >= self.n_states - 1:
            if action == 2:
                return state
        if (state + 1) % self.columns == 0:
            if action == 3:
                return state

        # If the action is valid then the resultant state is returned
        if action == 0:
            next_state = state - self.columns
        if action == 1:
            next_state = state - 1
        if action == 2:
            next_state = state + self.columns
        if action == 3:
            next_state = state + 1
        return next_state
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        return self.probs[next_state,state,action]
    
    def r(self, next_state, state, action):
        return self.rewards[next_state,state,action]
   
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))

################ Model-based algorithms ################
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.int)
    
    identity = np.identity(env.n_actions)
    # obtaining models
    p = env.probs
    r = env.rewards

    ini_iteration = 0
    stop = False

    while ini_iteration < max_iterations and not stop:
        dt = 0

        for s in range(env.n_states):
            ini_value = value[s]
            policy_action_prob = identity[policy[s]]
            value[s] = np.sum(policy_action_prob * p[:,s,:] * (r[:,s,:] + (gamma * value.reshape(-1, 1))))
            dt = max(dt, abs(ini_value - value[s]))

        ini_iteration += 1
        stop = dt < theta
        
    return value
    
def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int) 
    # obtaining models from the environment
    for state in range(env.n_states):
        alpha = float('-inf')
        correct_action = 0
        for action in range(env.n_actions):
            val = 0
            for ns in range(env.n_states):
                p = env.p(ns, state, action)
                r = env.r(ns, state, action)
                val += p * (r + gamma * value[ns])
            # It will be determined as the best feedback
            if alpha < val:
                alpha = val
                correct_action = action
        # we need to store feedback in the array.
        policy[state] = correct_action
    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    prev_policy = np.zeros(env.n_states, dtype=int) # we need to keep track of past policies in order to determine when there has been no improvement and, as a result, when the policy should be terminated.

    initial_iteration = 0

    while initial_iteration < max_iterations:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        initial_iteration += 1

        if np.all(np.equal(policy, prev_policy)): # if the previous value is equal to the new value, stop the algorithm.
            break
        else:
            prev_policy = policy

    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
       
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    policy = np.zeros(env.n_states, dtype=int)

    dt = abs(theta) + 1
    iti = 0 

    while dt > theta and max_iterations > iti:
        dt = 0

        for state in range(env.n_states):
            old_value = value[state]
            new_value = []

            for action in range(env.n_actions):
                total_exprected_return = 0

                for next_state in range(env.n_states):
                    next_state_probability = env.p(next_state, state, action)

                    disc_reward = env.r(next_state, state, action) + (gamma*value[next_state])

                    total_exprected_return += next_state_probability * disc_reward
                
                new_value.append(total_exprected_return)

            value[state] = max(new_value)

            dt = max(dt, np.abs(old_value - value[state]))
        
        iti += 1

    for state in range(env.n_states):
        new_actions = []
        new_action_values = []
        for action in range(env.n_actions):
            for next_state in range(env.n_states):
                next_state_probability = env.p(next_state, state, action=action)
                
                disc_reward = env.r(next_state, state, action=action) + (gamma*value[next_state])

                new_actions.append(action)
                new_action_values.append(next_state_probability*disc_reward)

        best_action = new_actions[new_action_values.index(max(new_action_values))]
        policy[state] = best_action

    print("The number of value iterations :-> ",iti)

    return policy, value

################ Tabular model-free algorithms ################
class epsilon_greedy_selection:
    def __init__(self, epsilon, random_state=None):
        self.random_state = random_state
        self.epsilon = epsilon

    def selection(self, q_s):
        if self.random_state.uniform(0, 1) < self.epsilon:
            return self.random_state.randint(0, len(q_s))
        else:
            best_action = np.max(q_s)
            best_action_index = np.flatnonzero(best_action == q_s)

            return self.random_state.choice(best_action_index)

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        epsilon_selection = epsilon_greedy_selection(epsilon[i], random_state)
        action = epsilon_selection.selection(q[s])

        for j in range(env.max_steps):
            next_state, reward, done = env.step(action)

            next_action = epsilon_selection.selection(q[next_state])

            q[s, action] = q[s, action] + (eta[i] * (reward + (gamma * q[next_state, next_action]) - q[s, action]))

            s = next_state
            action = next_action

            if done:
                break
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:

        epsilon_selection = epsilon_greedy_selection(epsilon[i], random_state)
        action = epsilon_selection.selection(q[s])
        
        for j in range(env.max_steps):
            next_state, reward, done = env.step(action)

            next_action = epsilon_selection.selection(q[next_state])

            q[s, action] = q[s, action] + (eta[i] * (reward + (gamma * np.max(q[next_state])) - q[s, action]))

            s = next_state
            action = next_action

            if done:
                break

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:

        state = features

        epsilon_selection = epsilon_greedy_selection(epsilon[i], random_state)
        action = epsilon_selection.selection(q)

        done = False
        while not done:
            next_state, reward, done = env.step(action)

            delta = reward - q[action]
            q = next_state.dot(theta)

            next_action = epsilon_selection.selection(q)

            theta += eta[i] * (delta + (gamma * q[next_action])) * state[action,:]

            state = next_state
            action = next_action
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

        q = features.dot(theta)

        state = features

        epsilon_selection = epsilon_greedy_selection(epsilon[i], random_state)
        action = epsilon_selection.selection(q)

        done = False
        while not done:
            next_state, reward, done = env.step(action)

            delta = reward - q[action]
            q = next_state.dot(theta)

            next_action = epsilon_selection.selection(q)

            theta += eta[i] * (delta + (gamma * np.max(q))) * state[action,:]

            state = next_state
            action = next_action

    return theta    

################ Main function ################

def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')
    
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    
    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

if __name__ == '__main__':
    main()
