### Author Name: Manqing Mao
### Luna Lander

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.101
        self.sample_size = 32
        self.train_start = 20000
        # create replay memory using deque
        self.memory = deque(maxlen=200000)
        self.train_buffer = deque(maxlen=10240)
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):

        #do training if buffer is full
        if len(self.train_buffer) == self.train_buffer.maxlen:
            batch_size = self.train_buffer.maxlen        
            states = np.zeros((batch_size, self.state_size))
            next_states = np.zeros((batch_size, self.state_size))
            action, reward, done = [], [], []

            for i in range(batch_size):
                states[i] = self.train_buffer[i][0]
                action.append(self.train_buffer[i][1])
                reward.append(self.train_buffer[i][2])
                next_states[i] = self.train_buffer[i][3]
                done.append(self.train_buffer[i][4])

            target = self.model.predict(states)            
            target_next = self.model.predict(next_states)
            target_val = self.target_model.predict(next_states)

            
            for i in range(batch_size):
                # Q Learning: get maximum Q value at s' from target model
                if done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    a = np.argmax(target_next[i])
                    target[i][action[i]] = reward[i] + self.discount_factor * (
                        target_val[i][a])
                    
            self.model.fit(states, target, 100, epochs=1, verbose=0)
            self.train_buffer.clear()

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
scores, episodes = [], []
action_count = 0

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    
    while not done:
        action = agent.get_action(state)        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
                    
        agent.append_sample(state, action, reward, next_state, done)
        
        if len(agent.memory) > agent.train_start:                
            action_count += 1
        
            if action_count == 4:
                mini_batch = random.sample(agent.memory, agent.sample_size)
                agent.train_buffer.extend(mini_batch)
                action_count = 0
                
            
        agent.train_model()        
        score += reward
        state = next_state
        
        if done:
            agent.update_target_model()
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            scores.append(score)
            episodes.append(e)
            print("episode:", e, " score:", score, " epsilon:", agent.epsilon," memory size:", len(agent.memory),"avg last 100:",np.mean(scores[-100:]))

#test model
test_scores = []

for e in range(100):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])
        
    while not done:
#         env.render()
        action = np.argmax(agent.model.predict(state)[0])        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])                                            
        score += reward
        state = next_state
        
        if done:
            test_scores.append(score)
            print("test episode:", e, " score:", score)



res = []
for i in range(1,len(scores)):
    j = 0 if i < 100 else i-100
    res.append(np.mean(scores[j:i]))
