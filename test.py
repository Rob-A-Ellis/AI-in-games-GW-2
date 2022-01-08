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
        
        # TODO:
        super(FrozenLake, self).__init__(n_states, n_actions, max_steps, pi, seed)

        self.probs = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)
        self.rewards = np.zeros((self.n_states, self.n_states, self.n_actions), dtype=float)

        self.calc_probs()
        self.calc_rewards()

        self.check_probs()

    def calc_probs(self):
        for state in range(self.n_states):
            if state == self.absorbing_state or self.lake_flat[state] in ('#','$'):
                self.probs[self.absorbing_state,state,:] = 1
                continue
            for action in range(self.n_actions):
                for slip_action in range(self.n_actions):
                    next_state = self.act(slip_action, state)
                    self.probs[next_state,state,action] += self.slip/self.n_actions
                    if action == slip_action:
                        self.probs[next_state,state,action] += 1 - self.slip

    def check_probs(self):
        target = np.load('p.npy')

        if self.probs.all() == target.all():
            print("Probs match target")
        else:
            print("Probs differ from target")

    def calc_rewards(self):
        for state in range(self.n_states):
            if state != self.absorbing_state:
                if self.lake_flat[state] == '$':
                    self.rewards[self.absorbing_state,state,:] = 1

    def act(self, action, state):
        _, self.columns = self.lake.shape

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
        
        if action == 0:
            new_state = state - self.columns
        if action == 1:
            new_state = state - 1
        if action == 2:
            new_state = state + self.columns
        if action == 3:
            new_state = state + 1
            
        return new_state
        
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

if __name__ == '__main__':
	seed = 0

	# Small lake
	lake = [['&', '.', '.', '.'],['.', '#', '.', '#'],['.', '.', '.', '#'],['#', '.', '.', '$']]

	env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

	play(env)