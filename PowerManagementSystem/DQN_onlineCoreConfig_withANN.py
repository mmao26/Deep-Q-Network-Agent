'''
Due to high complexity of the current DQN with experince replay, we 
say it is useful for offline learning. 
However, it doesn't appear to be great for online learning unless 
we simplify its implementation. Therefore, we call this version DQN_offline.py
This version is giving good results for the workloads
'''

import random
from collections import deque
from itertools import islice
from keras.models import Sequential, load_model

#from keras.layers import Dense
#from keras import optimizers
#from keras import regularizers
from datetime import datetime
#from keras.optimizers import Adam

import numpy as np
import pandas as pd                                                             # import pandas package and call it pd                                                             # import numpy package and call it np
import matplotlib.pyplot as plt                                                 # import plotting package and call it plt
pd.options.mode.chained_assignment = None    # default='warn'

np.random.seed(1)
random.seed(2)
from EnvGeneratorCoreConfig import WorkloadEnv
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
memResult = deque(maxlen=200) # Store the results in this memory deque

class DQNAgent:
    def __init__(self, state_size, action_size, output_size, memory_size):
        #self.state_size = state_size
        self.action_size = action_size
        #self.output_size = output_size
        self.memory = deque(maxlen=memory_size)
        self.reward_memory = deque(maxlen=memory_size)
        
        self.gamma = 0.1#0.95    # discount rate
        self.epsilon = 0.00001 # exploration rate
        self.epsilon_min = 0.00001 # Todo: Change this?
        self.epsilon_decay = 0.999#0.9999#0.99#1
        
        self.lr_Lowerbound = 0.001
        self.lr_Upperbound = 0.001
        self.learning_rate = self.lr_Lowerbound
        self.lr_decay = 0.1
        self.track_reward = [0,0,0,0]
        
        self.use_prioritized_exp_replay = True
        
        self.nn_input_dim = state_size  # input layer dimensionality
        self.nn_hdim0 = 14
        self.nn_hdim1 = 14
        self.nn_output_dim = output_size  # output layer dimensionality
        
        if(LOAD_PREV_MODEL):
            self.model_offline = load_model(open_file_name) # Need to test this functionality!
            self.model = build_model_w(self.model_offline.get_weights()) # Todo: Initialize the weights to the offline learned parameters
        else:
            self.model = build_model(self.nn_input_dim, self.nn_output_dim, self.nn_hdim0, self.nn_hdim1) # Todo: Initialize the weights to the offline learned parameters


    def reset_model(self):
        # Neural Net for Deep-Q learning Model
        #self.model_offline = load_model(open_file_name) # Need to test this functionality!
        self.model = build_model_w(self.model_offline.get_weights()) # Todo: Initialize the weights to the offline learned parameters
        return model
    
    # Make the Learning Rate Adaptive
    def adaptive_lr(self, reward, new_reward):
        reward_thld = -10
        reward += new_reward
        if(reward <= reward_thld):
            self.learning_rate = self.lr_Upperbound
            reward = 0
        else:
            self.learning_rate = max(self.lr_Lowerbound, self.learning_rate*self.lr_decay)
        return reward
    
    # Store the results in a queue
    def remember_result(self, nkey, chooseWl, output_trace,                         oracle_trace, Accuracy_PPW, optimalScore, score):
        self.memResult.append((nkey, chooseWl, output_trace, oracle_trace, Accuracy_PPW, optimalScore, score))

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
        #q_values = self.model.predict(state)                                    # Forward pass on the ANN
        q_values = predict(self.model, state) # Use custom ANN design 
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
        
#        if(2 in action):
#            print("DEBUG: Found 2", action)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay           
        
        return action, q_values  # returns action
    
    # Supports multiple actions (frequency and core configuration changes)
    def train_the_model(self, prev_state_q_val, prev_action, reward, prev_state, state):                  
        # Target Q value of state S using next state n_S           
        target_q = [0,0,0,0]
        for i in range(0,4):
            indL = 3*i
            indU = 3*i + 3
            #q_val_ns_a = self.model.predict(state)[0][indL: indU]
            q_val_ns_a = predict(self.model, state)[0][indL: indU] 
            target_q[i] = (reward[i] + self.gamma*np.amax(q_val_ns_a))
            #self.track_reward[i] = self.adaptive_lr(self.track_reward[i], reward[i])
            
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
        #self.model.fit(prev_state, target_q_cap, epochs=1, verbose=0)
        if(ONLINE_TRAINING):
            self.model = fit_model(state, target_q_cap, self.learning_rate, self.model, epoch=1, batch_size=1, print_loss=False)      # Update the model
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay         
            

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

if __name__ == "__main__":
    # Environment Generator
    allWl = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22]#[16]#[1, 2]  # Workloads to be chosen, "This should be a list from" --> [1:2, 5:16, 21, 22, 24, 25]    
    
    N_w = 10#3#10 # Number of workloads to choose
    
    N_k = 20 #3
    
    for nkey in range(0, N_k):                                                       # This for loop is a hack to generate random workloads 
        print("Key:", nkey)
        #chooseWl = random.sample(allWl, N_w) # Choose ~80% workloads
        
        chooseWl =  [21, 10, 13, 6, 8, 5, 12, 7, 22, 16] 
        #chooseWl =  [21, 11]# [21, 11, 7, 1, 15, 8, 10, 13, 9, 6] 

        print('Workloads Used for Testing: ', chooseWl)
        
        coreConfig = [1, 1]         # Set the number of little and big cores in the dataset
                                    # A15_freq_table = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]                         # A15 frequency table
                                    # num_freq = len(A15_freq_table)
                                    
        for i in range(np.size(chooseWl, 0)):
            model_name += '_' + str(chooseWl[i])
        model_name += '.h5' # File name (HDF5 file extension)
        
        env = WorkloadEnv(chooseWl, coreConfig, False)
    
        # Read the raw data
        df = pd.read_csv('data/sorted_all_data.csv', header=None)
    
        df_temp1 = env.normalizeHW(df)
        df_all_data, NumSnippets = env.f_setWorkload(df_temp1)
        
        #df_all_data = env.f_setCoreConfig(df_all_data) # Set a core configuration
        
        # Choose a few snippets for training out of these
    #    if(len(chooseWl) > 5):
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
        memory_size = 3000
        num_choices = int(len(df_env.index)/NumSnippets)  # 8 for WL16
        
        output_trace = np.zeros((NumSnippets, np.size(df_all_data, 1)))
    
        # Initializtion DQN
        agent = DQNAgent(state_size, action_size, output_size, memory_size)
        
        # Find pririty of snippets using the oracle trace
        Pr_priority = env.f_get_priority(oracle_trace, 5) # Todo: We can improve this function 
        
        scores, episodes = [], []
        
        batch_size = 100
        store_q_val16 = [] # Used for debugging
        store_q_val1 = [] # Used for debugging
        store_q_val_last = [] # Used for debugging
        
        config_cntr = 0 # If the config counter is more than a thld then we should occasionally increase frequency and cores
    
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
                debug_var = []
            
            while not done:
                
                # Take action based on off-policy 
                action, state_q_vals = agent.act(state)                                           
             
                
                # Debugging
    #            if(curr_snippet == 16):
    #                store_q_val16.append(agent.model.predict(state)[0])
    #            if(curr_snippet == 1):
    #                store_q_val1.append(agent.model.predict(state)[0])
    #            if(curr_snippet == NumSnippets):
    #                store_q_val_last.append(agent.model.predict(state)[0])
    #            
                # Interact with the environment
                oracle_ind, next_state, reward, done, debug_var = env.f_env(df_all_data, df_env, action, 
                                        state, prev_action, prev_state, curr_snippet, NumSnippets)
                
                score += sum(reward) 
                agent.reward_memory.append(reward)                    
    
                prev_action = action
                prev_state = state
                state = next_state
                curr_snippet += 1                                                   # Update the current snippet
                
                
    #            print("Snippet:", curr_snippet)
                if(done): # Episode finishes
                    scores.append(score)
                    episodes.append(e)
                    optimalScore = NumSnippets*4
                    print("episode:", e, " score:", score, " | epsilon:", agent.epsilon, " | avg last 100:",np.mean(scores))
                    
                    #print("optimal score:", optimalScore)
                    break
                
                output_trace[curr_snippet-1, :] = df_all_data.loc[oracle_ind, :]     # We do -1 because index starts from 0
                agent.train_the_model(state_q_vals, prev_action, reward, prev_state, state)
                  
    
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
        
        
        memResult.append((nkey, chooseWl, output_trace, oracle_trace, Accuracy_PPW, optimalScore, score))              
        
        print('Accuracy: ', Accuracy_PPW, '%')
    
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
    

#    l1,l2, l3 = plt.plot(store_q_val_last)
#    plt.ylabel('Snippet END q val')
#    plt.xlabel('Episodes')
#    plt.legend((l1,l2, l3), ('Decrease', 'Same', 'Increase'))
#    plt.show()
    
    print("Key:","Workloads:","Accuracy PPW:","% Wrong Actions")
    for nkey, chooseWl, output_trace, oracle_trace, Accuracy_PPW, optimalScore, score in memResult:
        perc_wrong_actions = (optimalScore - score)/optimalScore*100
        print(nkey,":", chooseWl,":", round(Accuracy_PPW,2),":", round(perc_wrong_actions,2))
        
    print('The entire code running time is ', datetime.now() - startTime)    
    
    # Save the model output
    if(SAVE_OUTPUT):
        agent.model.save(model_dir+model_name)
