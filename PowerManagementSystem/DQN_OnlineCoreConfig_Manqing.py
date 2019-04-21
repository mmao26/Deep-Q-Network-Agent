"""
@authors: Manqing Mao
@description: v1.0 of reinforcement learning on the DyPO data set in Python

"""

'''
Due to high complexity of the current DQN with experince replay, we 
say it is useful for offline learning. 
However, it doesn't appear to be great for online learning unless 
we simplify its implementation. Therefore, we call this version DQN_offline.py
This version is giving good results for the workloads
'''

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import numpy as np

np.random.seed(1)
random.seed(2)
#from tensorflow import set_random_seed
#set_random_seed(2)

from collections import deque
from itertools import islice
#from keras.models import Sequential, load_model
#from keras.layers import Dense
#from keras import optimizers
#from keras import regularizers
from datetime import datetime
#from keras.optimizers import Adam
#import tensorflow as tf

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import pandas as pd                                                             # import pandas package and call it pd                                                             # import numpy package and call it np
import matplotlib.pyplot as plt                                                 # import plotting package and call it plt
pd.options.mode.chained_assignment = None    # default='warn'

import EnvGeneratorCoreConfig
from ANN import *
startTime = datetime.now()                                                      # Get the start time of the code

EPISODES = 1

# SAVE THE OUTPUT MODEL
SAVE_OUTPUT = False
LOAD_PREV_MODEL = True
ONLINE_TRAINING = True
model_dir = 'models/' # Directory name
model_name = 'DQN_W'
#open_file_name = 'models/DQN_W_1.h5'
open_file_name = 'models/DQN_W_1_2_5_6_7_8_9_11_12_13_16_21_22.h5'

DEBUG = 1
memResult = deque(maxlen=50) # Store the results in this memory deque

class DQNAgent:
    def __init__(self, state_size, action_size, output_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.output_size = output_size
        self.memory = deque(maxlen=memory_size)
        self.Snippet_memory = deque(maxlen=memory_size)
        self.gamma = 0.1#0.95    # discount rate
        self.epsilon = 0.5 # exploration rate
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.96#0.9999#0.99#1
        self.learning_rate = 0.004 #0.001
        self.learning_rate_min = 0.0001 
        self.learning_rate_decay = 0.995
        self.use_prioritized_exp_replay = False
        
        self.nn_input_dim = state_size  # input layer dimensionality
        self.nn_hdim0 = 14
        self.nn_hdim1 = 14
        self.nn_output_dim = output_size  # output layer dimensionality
        """
        if(LOAD_PREV_MODEL):
            self.model_offline = load_model(open_file_name) # Need to test this functionality!
            self.model = build_model_w(self.model_offline.get_weights()) # Todo: Initialize the weights to the offline learned parameters
        else:
            self.model = build_model(self.nn_input_dim, self.nn_output_dim, self.nn_hdim0, self.nn_hdim1) # Todo: Initialize the weights to the offline learned parameters
        """
    """
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(14, input_dim=self.state_size, activation='relu'))      # Hidden layer
        model.add(Dense(14, activation='relu'))                                 # Hidden layer
        model.add(Dense(self.output_size, activation='linear'))                               # Output layer
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))        # default learning rate is 0.001
        return model
    """
    def remember(self, state_q_vals, state, action, reward, next_state, done):
        self.memory.append((state_q_vals, state, action, reward, next_state, done))
        
    def remember_snippet(self, curr_snippet):
        self.Snippet_memory.append((curr_snippet))

    def find_max(self, q_values):
        # If we have more than 1 q_value that are equal and maximum 
        # then we randomly choose one of those values
        max_q = np.max(q_values)
        ind_max_q = np.nonzero(q_values >= max_q)
        length_max_q = np.size(ind_max_q, 1)
        a_ix = np.random.randint(length_max_q)
        a = ind_max_q[0][a_ix]
        return a

    def act(self, state):
        # Output will be a vector of increase, decrease, keep same.
        # <num of little core, little freq, number of big cores, big freq>
        # Todo: Make this better by assigning independent random/Q value action to each
        # Currently either all actions are chosen from Q value or they are random. 
        # We should also suport any 2 as random and other 2 from q value
        action = [0,0,0,0]
        """
        q_values = self.model.predict(state) # Forward pass on the ANN
        """
        q_values = predict(model, state)
        if np.random.rand() <= self.epsilon:
            # Randomly choose between the action 0,1,2 in the vector
            for i in range(0,4):
                action[i] = random.randrange(self.action_size)
        else:    
            # Choose action based on Q-function
            for i in range(0,4):
                indL = 3*i
                indU = 3*i + 3
                action[i] = self.find_max(q_values[0][indL:indU])

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay           
        
        return action, q_values  # returns action
    
    def custom_sampling(self, batch_size, Pr_priority):
        if(self.use_prioritized_exp_replay):
            # Sample based on custom probability distribution without replacement
            #https://stats.stackexchange.com/questions/100709/choosing-elements-of-an-array-based-on-given-probabilities-for-each-element
            length = len(self.Snippet_memory)
            k = np.zeros((length, 1))
            u = np.random.rand(length, 1)
            cnt = 0
            for curr_snippet in self.Snippet_memory:
                #k[cnt] = u[cnt]**(1/Pr_priority[curr_snippet-1]) # Original formula
                k[cnt] = u[cnt]**(Pr_priority[curr_snippet-1])
                
                cnt += 1
            sorted_ind = np.argsort(k, axis=0) # sort index based in ascending order
            
            # Pick larget k values. Items will be equal to batch size
            trunc_sorted_ind = sorted_ind[-batch_size:]
            # Inefficient code, right now focus on functionailty. 
            minibatch = []
            mb_test = []
            for i in range(0, batch_size):
                ind = trunc_sorted_ind[i]
                minibatch.append(self.memory[ind[0]])
                mb_test.append(self.Snippet_memory[ind[0]])
            #minibatch = deque(islice(self.memory, sorted_ind[-batch_size-1], sorted_ind[-1]))
        else:
            # Randomly sample training data from memory
            minibatch = random.sample(self.memory, batch_size)  
            mb_test = []

            
        return minibatch, mb_test
        
        

    def replay(self, batch_size, Pr_priority, model):
        
        minibatch, mb_test = self.custom_sampling(batch_size, Pr_priority)
        
        for prev_state_q_vals, prev_state, prev_action, reward, state, done in minibatch:               # For each row in training data
            model = self.train_the_model(prev_state_q_vals, prev_action, reward, prev_state, state, model)    
        return model
    
    def train_the_model(self, prev_state_q_val, prev_action, reward, prev_state, state, model):                  
        # Target Q value of state S using next state n_S           
        target_q = [0,0,0,0]
        for i in range(0,4):
            indL = 3*i
            indU = 3*i + 3
            """
            q_val_ns_a = self.model.predict(state)[0][indL: indU]
            """
            q_val_ns_a = predict(model, state)[0][indL: indU]
            target_q[i] = (reward[i] + self.gamma*np.amax(q_val_ns_a))          # TD0
            #target_q[i] = np.amax(q_val_ns_a)                                    # Monte-Carlo
            
        # This is the predicted value of Q function for a given state S
        target_q_cap = prev_state_q_val                               
            
        # Replace the target_q_cap for the current action with the true value, 
        # keeping other values same.
        # This will have an effect that the error is only due to the current action and not other actions
        #a_nl = action[0]
        #a_fl = action[1]
        #a_nb = action[2]
        #a_fb = action[3]
        for i in range(0,4):
            target_q_cap[0][prev_action[i] + 3*i] = target_q[i]
        
        # Fit the data and adapt the model
        """
        self.model.fit(prev_state, target_q_cap, epochs=1, verbose=0)
        """
        model = fit_model(prev_state, target_q_cap, self.learning_rate, model, epoch=1, batch_size=1, print_loss=False) 
        return model
    
def f_randomize_snippets(data):
        
    # Find the number of snippets
    numSnippets = np.max(data.loc[:, 0])
    data_length = np.size(data, 0)
    numConfigs = int(data_length/numSnippets)
    # Form an array and randomize it from 1 to total snippets
    snippet_vec = np.arange(1, numSnippets+1, dtype='int')
    np.random.shuffle(snippet_vec)
    # Repmat number of configuration times and then reshape into vector again
    snippet_vec = np.tile(snippet_vec, (numConfigs, 1))
    snippet_vec = snippet_vec.reshape(data_length, 1, order='F')
    # Now replace the snippet column with the new vector
    data.loc[:, 0] = snippet_vec
    # Then sort w.r.t snippet column
    data = data.sort_values(by=[0, 33], ascending=[True, True])
    data = data.reset_index(drop=True)

    return data

#class sliceable_deque(deque):
#    def __getitem__(self, index):
#        try:
#            return deque.__getitem__(self, index)
#        except TypeError:
#            return type(self)(islice(self, index.start,
#                                               index.stop, index.step))
#

if __name__ == "__main__":
    # Environment Generator
    #chooseWl = [1]#[21, 22]#[21]#[1]                 # Workloads to be chosen, "This should be a list from" --> [1:2, 5:16, 21, 22, 24, 25]
    #chooseWl = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22]#[16]#[1, 2]  # Workloads to be chosen, "This should be a list from" --> [1:2, 5:16, 21, 22, 24, 25]
    #chooseWl = np.sort(random.sample(chooseWl, 13)) # Choose ~80% workloads
    chooseWl = [ 1,  2,  5,  6,  7,  8,  9, 11, 12, 13, 16, 21, 22] # Already chosen randomly as 80% of workloads
    #chooseWl = [2,5,7,8,12,13,21,22] # Already chosen randomly as 50% of workloads randomly

    print('Workloads used for training: ', chooseWl)
    
    coreConfig = [1, 1]         # Set the number of little and big cores in the dataset
                                # A15_freq_table = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]                         # A15 frequency table
                                # num_freq = len(A15_freq_table)
                                
    for i in range(np.size(chooseWl, 0)):
        model_name += '_' + str(chooseWl[i])
    model_name += '.h5' # File name (HDF5 file extension)
    
    env = EnvGeneratorCoreConfig.WorkloadEnv(chooseWl, coreConfig, False)

    # Read the raw data
    df = pd.read_csv('data/sorted_all_data.csv', header=None)

    df_temp1 = env.normalizeHW(df)
    df_all_data, NumSnippets = env.f_setWorkload(df_temp1)
    
    #df_all_data = env.f_setCoreConfig(df_all_data) # Set a core configuration
    
    # Choose a few snippets for training out of these
    #if(len(chooseWl) > 11):
#        df_all_data, NumSnippets = env.f_find_training_snippets(df_all_data)

    # Generate the oracle trace
    start_config = [4, 1.4, 4, 2]                                               # Start configuration [num of little cores (4),little core freq (1.4GHz), num of big cores (4), Highest frequency (2 GHz)]
    oracle_trace = env.oracleTrace(NumSnippets, df_all_data, start_config) 
    
    
    # Create the feature data
    df_temp4 = env.f_createFeatureSet(df_all_data)
    
    # Normalize the feature data
    df_env = env.f_normalize_data(df_temp4)
    
    state_size = len(df_env.columns)     # Number of Features
    action_size = 3
    output_size = 12
    memory_size = 2250 # 3000 # snippets are 2247
    k = 2                                                                       # Prioritizing the rewards being +ve or -ve
    num_choices = int(len(df_env.index)/NumSnippets)  # 8 for WL16
    
    output_trace = np.zeros((NumSnippets, np.size(df_all_data, 1)))

    # Initializtion DQN
    agent = DQNAgent(state_size, action_size, output_size, memory_size)
    
    # Find pririty of snippets using the oracle trace
    Pr_priority = env.f_get_priority(oracle_trace, 0.25) # Todo: We can improve this function 
    
    scores, episodes = [], []

    # Initilize model
    model = build_model(agent.nn_input_dim, agent.nn_output_dim, agent.nn_hdim0, agent.nn_hdim1)
    
    batch_size = 100
    store_q_val16 = []                                                          # Used for debugging
    store_q_val1 = []                                                           # Used for debugging
    store_q_val_last = []                                                       # Used for debugging
    break_me  = False
    train_cntr = 0                                                              # This counter increments when not training and then reset when we train. This is to run training every 300 samples. 
    delta = 0                                                                   # Use to change the probabilites of +ve and -negative reward
    for e in range(EPISODES):

        done = False                                                            # True means episode has finished

        curr_snippet = 1
        score = 0
        ind_config = env.findStateIx(df_all_data, start_config, curr_snippet)     # Given an index to choose state
        output_trace[0, :] = df_all_data.values[ind_config, :]
        state = df_env.values[ind_config, :]
        
        prev_action = [1, 1, 1, 1]                                              # Start from arbitary point [This is unknown]
        prev_state = state                                                      # Start from arbitary point [This is unknown]
        #prev_freq = curr_freq                                                  # Start from arbitary point [This is unknown]
        
        if(DEBUG):
            #debug_var = np.zeros((NumSnippets, 4))
            store_debug_var = []
            
        # Update the reward counters   
        if(delta > 0):
            print("Debug")
        num_p_reward = 0
        num_n_reward = 0
        Pr_p_reward = 0.5 - delta
        Pr_n_reward = 0.5 + delta
        
        while not done:
            
            # Take action based on off-policy 
            action, state_q_vals = agent.act(state)                                           
         
#            # Debugging
#            if(curr_snippet == 16):
#                store_q_val16.append(agent.model.predict(state)[0])
#            if(curr_snippet == 1):
#                store_q_val1.append(agent.model.predict(state)[0])
#            if(curr_snippet == NumSnippets):
#                store_q_val_last.append(agent.model.predict(state)[0])
            
            # Interact with the environment
            oracle_ind, next_state, reward, done, debug_var = env.f_env(df_all_data, df_env, action, 
                                    state, prev_action, prev_state, curr_snippet, NumSnippets)
            #if(e == 165):
            # Store for Debugging: [snippet_name, selected_action, optimal_actions, rewards]
            store_debug_var.append((curr_snippet, debug_var[1], debug_var[3], debug_var[-1]))
            
            # IMP: For debugging, print this store_debug_var then, 
            # compare with optimal configurations from Matlab
            # i = 0; max_ppw = max(sorted_all_data_temp(i*128 + 1: (i+1)*128, ind_PPW))
            #sorted_all_data_temp((sorted_all_data_temp(i*128 + 1: (i+1)*128, ind_PPW) == max_ppw),[ind_freq, ind_A15, ind_A7])
                        
            
            
            
            # Store the data in memory for replay later
            # The first time we have dummy prev state and action, so we skip to save the first value
            if(curr_snippet > 1):
                
                # Sum of reward in current interval
                sum_r = sum(reward)
                score += sum_r 
                
                add_to_memory = 0
                rand_num = np.random.rand(1)[0]
                if(sum_r > 0):
                    num_p_reward += 1
                    if(rand_num > Pr_p_reward):
                        add_to_memory = 1
                else:
                    num_n_reward += 1
                    if(rand_num > Pr_n_reward):
                        add_to_memory = 1
                    
                
                # We should see balanced number of samples 
                # So we put samples with a total negative reward into the memroy more                
                
                if((np.random.rand(1)[0] > Pr_priority[curr_snippet-1]) and add_to_memory):
                    agent.remember(state_q_vals, prev_state, prev_action, reward, state, done)              
                    agent.remember_snippet(curr_snippet)
                
                           
            
#            if(DEBUG):
#                #debug_var[curr_snippet-1] = [prev_state[0][-1], prev_action, state[0][-1], reward]
#                if(done):
#                    print(debug_var)    
#                else:
#                    debug_var_prev = np.concatenate((prev_state[0][11:15], np.array(prev_action)))
#                    debug_var_nxt = np.concatenate((state[0][11:15], np.array([reward])))
#                    debug_var = np.concatenate((debug_var_prev, debug_var_nxt))
#                    if(curr_snippet > 1):
#                        debug_var2 = np.stack((debug_var2, debug_var))
#                    else:
#                        debug_var2 = debug_var
                
            
            prev_action = action
            prev_state = state
            state = next_state
            curr_snippet += 1                                                   # Update the current snippet
            
            train_cntr += 1                                                     # Increment the training counter
            
            # When we have atleast batch_size elements in the agent.memory then 
            # do experince replay after every train counter samples
            if(train_cntr > 300):
                # reset the train counter
                train_cntr = 0
                #print("Training_now")
                
                if (len(agent.memory) > batch_size):
                             
                    # This replays the memory and performs ANN training
                    """
                    agent.replay(batch_size, Pr_priority)
                    """
                    model = agent.replay(batch_size, Pr_priority, model)
                    
                    if agent.learning_rate > agent.learning_rate_min:
                        agent.learning_rate = agent.learning_rate*agent.learning_rate_decay
                    if agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
            
            
            if(done): # Episode finishes
                scores.append(score)
                episodes.append(e)
                print("episode:", e, " score:", score, " | epsilon:", agent.epsilon, " | learning rate:", agent.learning_rate, " | avg last 100:",np.mean(scores[-100:]))
                #if(score >= 954):
                #    break_me = True
                
                # Assign probabilities of pushing +ve or -ve reward
                total_samples = num_p_reward + num_n_reward
                delta = (num_p_reward - (total_samples/2))/(total_samples*2)
                
                break
            
            output_trace[curr_snippet-1, :] = df_all_data.loc[oracle_ind, :]     # We do -1 because index starts from 0
            
            # Train the model
            #agent.train_the_model(state_q_vals, prev_action, reward, prev_state, state) 

        # End of while not done:
        
        #if(break_me):
        #    break
        
            
        
       

    # Keep track of cumulative mean of the score            
    cum_mean_score = [np.mean(scores[:i+1]) for i in range(np.size(scores)-1)]       
    
    #print('Accuracy: ', 100-(NumSnippets-cum_mean_score[-1]-1)/(NumSnippets-1)/2*100, '%')
    

    
    # Plot the data
   # Plot the data
    plt.plot(oracle_trace[:, env.col["fb"]], 'r^-', label="Oracle")
    plt.plot(output_trace[:, env.col["fb"]], 'b^-', label="RL: DQN")
    plt.ylabel('CPU Frequency (GHz)')
    plt.xlabel('Snippets')
    plt.legend()
    plt.show()
    
    plt.plot(oracle_trace[:, env.col["nl"]], 'r^-', label="Oracle")
    plt.plot(output_trace[:, env.col["nl"]], 'b^-', label="RL: DQN")
    plt.ylabel('Number of Little Cores Active')
    plt.xlabel('Snippets')
    plt.legend()
    plt.show()
    
    plt.plot(oracle_trace[:, env.col["nb"]], 'r^-', label="Oracle")
    plt.plot(output_trace[:, env.col["nb"]], 'b^-', label="RL: DQN")
    plt.ylabel('Number of Big Cores Active')
    plt.xlabel('Snippets')
    plt.legend()
    plt.show()
    
    plt.plot(oracle_trace[:, env.col["PPW"]], 'r^-', label="Oracle")
    plt.plot(output_trace[:, env.col["PPW"]], 'b^-', label="RL: DQN")
    plt.ylabel('PPW (Instr/nJ)')
    plt.xlabel('Snippets')
    plt.legend()
    plt.show()
    
    Accuracy_PPW = 100 - env.mean_absolute_percentage_error(oracle_trace[:, env.col["PPW"]],
                                                            output_trace[:, env.col["PPW"]])
    print('Accuracy: ', Accuracy_PPW, '%')

    plt.plot(scores, 'r^-')
    plt.ylabel('Total Reward')
    plt.xlabel('Episodes')
    plt.show()
    
    plt.plot(cum_mean_score, 'r^-')
    plt.ylabel('CUM Mean Total Reward')
    plt.xlabel('Episodes')
    plt.show()
    
#    l1,l2, l3 = plt.plot(store_q_val16)
#    plt.ylabel('Snippet 16 q val')
#    plt.xlabel('Episodes')
#    plt.legend((l1,l2, l3), ('Decrease', 'Same', 'Increase'))
#    plt.show()
#
#    l1,l2, l3 = plt.plot(store_q_val1)
#    plt.ylabel('Snippet 1 q val')
#    plt.xlabel('Episodes')
#    plt.legend((l1,l2, l3), ('Decrease', 'Same', 'Increase'))
#    plt.show()
#    print('The entire code running time is ', datetime.now() - startTime)    
#    
#
#    l1,l2, l3 = plt.plot(store_q_val_last)
#    plt.ylabel('Snippet END q val')
#    plt.xlabel('Episodes')
#    plt.legend((l1,l2, l3), ('Decrease', 'Same', 'Increase'))
#    plt.show()
#    print('The entire code running time is ', datetime.now() - startTime)    
    
    # Save the model output
    if(SAVE_OUTPUT):
        agent.model.save(model_dir+model_name)
