"""
Created on Fri Feb 23 11:00:19 2018

@authors: Manqing Mao
@modified by: Ujjwal Gupta

@description: v1.0 of reinforcement learning on the DyPO data set in Python

"""
#Import libraries
import math
import pandas as pd                                                             # import pandas package and call it pd
import numpy as np                                                              # import numpy package and call it np
import matplotlib.pyplot as plt                                                 # import plotting package and call it plt
pd.options.mode.chained_assignment = None    # default='warn'
np.random.seed(1)

class WorkloadEnv:
    def __init__(self, chooseWl, coreConfig):

        # Get workload and core configurations:
        self.chooseWl = chooseWl
        self.coreConfig = coreConfig

        self.A15_freq_table = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.A7_freq_table = [0.6, 0.8, 1, 1.2, 1.4]
        self.nl_table = [1, 2, 3 , 4]
        self.nb_table = [1, 2, 3 , 4]
        
        #self.obj = "PPW" # The objective name as per the column list below
        #self.obj_label = "PPW (Instr/nJ)" # Label of the objective function
        #self.obj_dir = 1 # 1: Maximize, 0: Minimize the objective
        
        self.obj = "EDP" # The objective name as per the column list below
        self.obj_label = "EDP (Energy-s)" # Label of the objective function
        self.obj_dir = 0 # 1: Maximize, 0: Minimize the objective
        
        
        self.objective_range = 0.01 # 1% lower objective is acceptable
        
        self.delta_freq = 0.2
        # Data index representation
        # Index name
        # Use: cname.index("snippet_name") for indexing..
        # Total of columns: 42
        self.col_name_lst = ["snippet_name"     ,                      # Name of the snippet
                             "t_exe"            ,                      # Execution time of a snippet
                                                          # Hardware counter columns
                             "num_instr"        , "num_cycles"         , "num_br_mispred"       ,
                             "num_L2_misses"    , "num_data_mem_access", "num_non_cache_mem_req",
                                                          # Power consumption and temperature
                             "cpu_power"        , "A7_power"           , "A15_power"            , "mem_power"    ,
                             "gfx_power"        , "temp1"              , "temp2"                , "temp3"        , "temp4"   , "temp5"   ,
                             "total_chip_power" ,
                                                          # Derived metrics like energy, PPW
                             "IPS"              , "PPW"                , "energy"               , "EDP"          , "ED2P"    , "EPI"     ,
                                                          # CPU utilizations
                             "A7util1"          , "A7util2"            , "A7util3"              , "A7util4"      , "A15util1", "A15util2",
                             "A15util3"         , "A15util4"           ,
                             "fb"             , "nl"                , "nb"                 , "workload_name",
                             "sum_util_A7"      , 
                             "sorted_util_A15_1", "sorted_util_A15_2"  , "sorted_util_A15_3"    , "sorted_util_A15_4", "fl"]
        
        """Enumerated column dictionary:
        col = {'snippet_name': 0, 't_exe': 1, ...'sorted_util_A15_4': 41}"""
        self.col = {k: v for v, k in enumerate(self.col_name_lst)}
        
        # Set of features to be selected
        self.feature_names = ["num_cycles"       , "num_br_mispred"     ,
                              "num_L2_misses"    , "num_data_mem_access", "num_non_cache_mem_req", "total_chip_power" , 
                              "sum_util_A7"      , 
                              "sorted_util_A15_1", "sorted_util_A15_2"  , "sorted_util_A15_3"    , "sorted_util_A15_4", 
                              "nl", "fl", "nb", "fb" ]
        
        self.f_name = {k: v for v, k in enumerate(self.feature_names)}

    # Normalization of feature data
    def f_normalize_data(self, data):    
        
        data_length = np.size(data, 0) # Number of data points

        min_data = np.array([0.496817233920306,	
                         6.33378921885967e-05,	
                         0.0210775248435517,	
                         0.0816601165551269,	
                         1.25473605828546e-06,
                         0, #0.40924,
                         0,
                         0, 0, 0, 0])
            
        max_data = np.array([6.20156632502085,	
                         0.0480556970623196,	
                         0.286624203821656,	
                         0.500697491048239,	
                         0.161657327220241,	
                         8.33805900000000,
                         100.0,
                         100.0, 100.0, 100.0, 100.0])
        
        range_data = max_data - min_data
        feat_data_ind = data.columns
        for i in range(0, np.size(range_data, 0)):
            data[feat_data_ind[i]] = data[feat_data_ind[i]].apply(lambda x: (x - min_data[i])/range_data[i])
           
        return data

        
    def normalizeHW(self, dataframe):
        # Store the original number of instructions
        num_instr = dataframe[[self.col["num_instr"]]]

        # Normalize the hardware counters w.r.t. number of instructions retired
        df2 = dataframe.copy()
        for i in range(self.col["num_instr"], self.col["num_non_cache_mem_req"]+1):
            df2.loc[:,i] = df2.loc[:, i] / dataframe.loc[:, self.col["num_instr"]]
        dataframe = df2                # Transfer data to the first data frame
        del df2                 # Delete the copy
        return dataframe
               
    """ Function to choose a specific set of workloads """
    def f_setWorkload(self, dataframe):       
        # Select the workloads that are in vector chooseWl
        ind = self.col["workload_name"]
   
        #df_temp = dataframe.loc[dataframe.loc[:,ind].isin(self.chooseWl)]
        df_temp = []
        for i in range(0, len(self.chooseWl)):
            df_temp.append(dataframe.loc[dataframe.loc[:,ind].isin([self.chooseWl[i]])])
        df_temp = pd.concat(df_temp)
        
        # Snippet Names for the specific workload
        snippet_name = df_temp.loc[:, self.col["snippet_name"]]
        
        configs = df_temp.loc[:, [self.col["nl"], self.col["fl"], self.col["nb"], self.col["fb"]] ]

        total_len = len(snippet_name)
        
        num_confis = len(np.unique(configs, axis=0)) # the number of choices for one snippet -- E.g., 128
        
        num_snippets = int(total_len/num_confis) # the number of snippets in one workload -- E.g., 248
   
        list_snippet_names = np.linspace(1, num_snippets, num_snippets)
        mod_snippet_names = np.tile(list_snippet_names, (num_confis, 1))	# Modified snippet names
   
        # Restructure the snippet names into an array
        new_snippet_names = mod_snippet_names.reshape(total_len, 1, order='F').copy()
   
        df_temp.loc[:, self.col["snippet_name"]] = new_snippet_names

        df_temp = df_temp.reset_index(drop=True)
        
        return df_temp, num_snippets;


    """ Function to choose a core configuration """
    def f_setCoreConfig(self, dataframe):

        # Select the workloads that are in vector chooseWl
        df_temp = dataframe.loc[dataframe.loc[:, self.col["nl"]] == self.coreConfig[0]]
        df_temp = df_temp.loc[df_temp.loc[:, self.col["nb"]] == self.coreConfig[1]]

        df_temp = df_temp.reset_index(drop=True)
        
        # We should also change the available big and little core table
        self.nl_table = [self.coreConfig[0]]
        self.nb_table = [self.coreConfig[1]]
        
        return df_temp;

    """ Function to create a feature dataset """
    def f_createFeatureSet(self, dataframe):
        # List of feature indices selected from the dictionary "col" with a for loop
        feature_indices = [self.col.get(key) for key in self.feature_names]
        df_temp = dataframe.loc[:, feature_indices]
        
        df_temp = df_temp.reset_index(drop=True)
        
        return df_temp;

    """ Generate the oracle trace """
    def oracleTrace(self, numSnippets, dataframe, start_config):
        
        oracle_trace = np.zeros((numSnippets, np.size(dataframe, 1)))
        
        curr_snippet = 1
        ind_start_config = self.findStateIx(dataframe, start_config, curr_snippet)
        oracle_trace[curr_snippet-1,:] = dataframe.loc[ind_start_config, :]
                
        for i in range(1, numSnippets):
            curr_snippet = i+1
            ind_snippet = dataframe.loc[:, self.col["snippet_name"]] == curr_snippet
            candidate_obj = dataframe.loc[ind_snippet, self.col[self.obj]]
    
            # Find the index of the best PPW for the next snippet
            if(self.obj_dir == 1):
                ind_best_obj = candidate_obj.idxmax()
            else:
                ind_best_obj = candidate_obj.idxmin()
    
            # Choose the best PPW row as the next row in oracle
            oracle_trace[i,:] = dataframe.loc[ind_best_obj, :]

        
        return oracle_trace
    
    def f_change_config_ind(self, action, curr_ind, max_val):
        # Apply the action:            
        if(action == 0):
            new_ind = curr_ind-1                                                # Go down one index
            new_ind = max(new_ind, 0)                                           # Apply lower bound 
        elif(action == 2):                                         
            new_ind = curr_ind+1                                                # Go up one index
            new_ind = min(new_ind, max_val)                                     # Apply upper bound
        else:                                          
            new_ind = curr_ind                                                  # No change
        return new_ind
    
    """   
    # action 0: Make the frequency/CoreConfig smaller
    # action 1: Frequency/CoreConfig remains same
    # action 2: Make the frequency/CoreConfig larger
    """
    def f_action_result(self, state, action):
        
        curr_config = self.findConfig(state)
       
        new_config = [0,0,0,0] 
        # For each configuration knob
        for i in range(0, 4):
            if(i == 0):
                curr_ind = self.nl_table.index(curr_config[i])                  # Get the index of the number of little cores
                max_val = len(self.nl_table) - 1
                new_ind = self.f_change_config_ind(action[i], curr_ind, max_val)
                new_config[i] = self.nl_table[new_ind]
            elif(i == 1):
                curr_ind = self.A7_freq_table.index(curr_config[i])                  # Get the index of the frequency of little cores
                max_val = len(self.A7_freq_table) - 1
                new_ind = self.f_change_config_ind(action[i], curr_ind, max_val)
                new_config[i] = self.A7_freq_table[new_ind]
            elif(i == 2):
                curr_ind = self.nb_table.index(curr_config[i])                  # Get the index of the num of big cores
                max_val = len(self.nb_table) - 1
                new_ind = self.f_change_config_ind(action[i], curr_ind, max_val)
                new_config[i] = self.nb_table[new_ind]
            else:
                curr_ind = self.A15_freq_table.index(curr_config[i])                  # Get the index of the frequency for big core
                max_val = len(self.A15_freq_table) - 1
                new_ind = self.f_change_config_ind(action[i], curr_ind, max_val)
                new_config[i] = self.A15_freq_table[new_ind]
        
        # Additional constraint in our dataset is that big and little core frequency are same
        # This will not be added when we implement it in hardware
        if (new_config[1] != new_config[3]):
            new_config[1] = min(new_config[3], max(self.A7_freq_table))
        
        return new_config
    
    def f_optimal_action_gen(self, dataframe, ind_best_obj_rows, ind_cols, prev_config_val):
        optimal_config_val = dataframe.loc[ind_best_obj_rows, ind_cols]
        
        # Check if the current frequency index is similar to the optimal frequencies or not
        # If they are smaller then we need to reduce the current frequency with action 0
        # Otherwise, we may have to increase or keep same. 
        optimal_action = []
        for i in optimal_config_val:
            if (i < prev_config_val):
                optimal_action.append(0)
            elif (i > prev_config_val):
                optimal_action.append(2)
            else:
                optimal_action.append(1)
        
        return optimal_action, optimal_config_val

    def f_reward(self, dataframe, prev_action, prev_state, curr_snippet):
    
        prev_config = self.findConfig(prev_state)#[0][-1]
        
        ind_curr_snippet = dataframe.loc[:, self.col["snippet_name"]] == curr_snippet
        candidate_obj = dataframe.loc[ind_curr_snippet, self.col[self.obj]]
    
        # Find the index of the best PPW for the current snippet
        #ind_best_obj = candidate_obj.idxmax()
        
        # Find the best value of the objective
        if(self.obj_dir == 1):
            best_obj = candidate_obj.max() # Just use max.
            # Range of good objective values:
            best_obj_range = best_obj*np.array([1-self.objective_range, 1])         
            # Find the indices that satisfy the optimal objective range
            ind_best_obj_LB = np.where(candidate_obj >= best_obj_range[0])[0]
        else:
            best_obj = candidate_obj.min() # Just use min.
            # Range of good objective values:
            best_obj_range = best_obj*np.array([0, self.objective_range])         
            # Find the indices that satisfy the optimal objective range
            ind_best_obj_LB = np.where(candidate_obj <= best_obj_range[0])[0]
        
        # Find the optimal action
        oa_nl, opt_config_nl = self.f_optimal_action_gen(dataframe, ind_best_obj_LB, self.col["nl"], prev_config[0])
        oa_fl, opt_config_fl = self.f_optimal_action_gen(dataframe, ind_best_obj_LB, self.col["fl"], prev_config[1])
        oa_nb, opt_config_nb = self.f_optimal_action_gen(dataframe, ind_best_obj_LB, self.col["nb"], prev_config[2])
        oa_fb, opt_config_fb = self.f_optimal_action_gen(dataframe, ind_best_obj_LB, self.col["fl"], prev_config[3])
        
        # If the current action is equal to any of the optimal actions then we have reward = 1
        reward = [0,0,0,0]
#        reward[0] = 1 if (any(oa_nl) == prev_action[0]) else -1
#        reward[1] = 1 if (any(oa_fl) == prev_action[1]) else -1
#        reward[2] = 1 if (any(oa_nb) == prev_action[2]) else -1
#        reward[3] = 1 if (any(oa_fb) == prev_action[3]) else -1
        
        reward[0] = 1 if (prev_action[0] in oa_nl) else -1
        reward[1] = 1 if (prev_action[1] in oa_fl) else -1
        reward[2] = 1 if (prev_action[2] in oa_nb) else -1
        reward[3] = 1 if (prev_action[3] in oa_fb) else -1
        
        return reward
        #return reward, oa_nl, oa_fl, oa_nb, oa_fb
    

    def findStateIx(self, df_total, config, snippet):
        
        # Find the current snippet index, then next freq index then take and
        ind_snippet = df_total.loc[:, self.col["snippet_name"]] == snippet
        
        # Configurations
        ind_nl = df_total.loc[:, self.col["nl"]] == config[0]
        ind_fl = df_total.loc[:, self.col["fl"]] == config[1]
        ind_nb = df_total.loc[:, self.col["nb"]] == config[2]
        ind_fb = df_total.loc[:, self.col["fb"]] == config[3]
        
        ind_snippet_config = np.logical_and(ind_snippet, ind_nl) # Find logical AND of the two
        ind_snippet_config = np.logical_and(ind_snippet_config, ind_fl) # Find logical AND of the two
        ind_snippet_config = np.logical_and(ind_snippet_config, ind_nb) # Find logical AND of the two
        ind_snippet_config = np.logical_and(ind_snippet_config, ind_fb) # Find logical AND of the two

        #state = df_feature.values[ind_snippet_config, :]
        return ind_snippet_config
    
    def findConfig(self, state):
                
        # Configurations
        config = [0, 0, 0, 0]
        config[0] = state[0][self.f_name["nl"]]
        config[1] = state[0][self.f_name["fl"]]
        config[2] = state[0][self.f_name["nb"]]
        config[3] = state[0][self.f_name["fb"]]
        return config
    
    #  
    def f_env(self, df_total, df_feature, action, state, 
              prev_action, prev_state, curr_snippet, numSnippets):
              
        # Compute the reward
        
        reward = self.f_reward(df_total, prev_action, prev_state, curr_snippet)
        
        # Apply the next action
        nxt_config = self.f_action_result(state, action)
        new_snippet = curr_snippet + 1
        
        ind_next_snippet_config = self.findStateIx(df_total, nxt_config, new_snippet)
        if(curr_snippet == numSnippets):
            done = True    
            next_state = None
        else:
            done = False
            next_state = df_feature.values[ind_next_snippet_config, :]
        
        return ind_next_snippet_config, next_state, reward, done

    # Finds the state (features) based on the frequency index
    # Todo: Find its utility
    #def IndextoState(self, dataframe, index, state_size):
        #state = dataframe.values[index, :]
        #return state
#        return np.reshape(state, [1, state_size])
    

    
    # Functin evaluates mean absolute percentage error between two 
    # numpy arrays
    # We use this to find the error in PPW
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  
    
    
    # This function chooses 80% of snippets for traning. 
    def f_find_training_snippets(self, data):
        
        # Find the number of configurations
        
        configs = data.loc[:, [self.col["nl"], self.col["fl"], self.col["nb"], self.col["fb"]]]

        totalLen = np.size(data, 0)
        
        numConfigs = len(np.unique(configs, axis=0)) # the number of choices for one snippet -- E.g., 128
        
        
        # Find the number of snippets
        numSnippets = int(totalLen/numConfigs)
        #data_length = np.size(data, 0)
        # Form an array and randomize it from 1 to total snippets
        snippet_vec = np.arange(1, numSnippets+1, dtype='int')
        np.random.shuffle(snippet_vec)
        
        TRAIN_PERC = 80 # Percentage
        train_snippets = snippet_vec[:int(numSnippets*TRAIN_PERC/100)]
        
        training_data = data.loc[data.loc[:,0].isin(train_snippets)]
    
        # Repmat number of configuration times and then reshape into vector again
        new_numSnippets = len(train_snippets)
        snippet_vec = np.arange(1, new_numSnippets+1, dtype='int')
        snippet_vec = np.tile(snippet_vec, (numConfigs, 1))
        data_length = new_numSnippets*numConfigs
        snippet_vec = snippet_vec.reshape(data_length, 1, order='F')
        # Now replace the snippet column with the new vector
        training_data.loc[:, 0] = snippet_vec
        # Then sort w.r.t snippet column
        training_data = training_data.sort_values(by=[0, 33], ascending=[True, True])
        training_data = training_data.reset_index(drop=True)
    
        return training_data, new_numSnippets

    def f_get_priority(self, oracle_trace, k):
        #p_i = 1 - exp(-k*n_i/nT) -> priority for snippet i
        #sampling probability: P_i = p_i/sum(p_i)
        # k is the priority parameter. 
        configs = oracle_trace[:, [self.col["nl"], self.col["fl"], self.col["nb"], self.col["fb"]]]
        
        n_T = np.size(oracle_trace, 0) # Number of snippets
        
        uniqueConfigs = np.unique(configs, axis=0, return_index=True, return_inverse=True, return_counts=True)
        
        #numConfigs = len(uniqueConfigs[0]) # Total number of unique configurations in the oracle trace
        
        # Reconstruction vector
        reconst = uniqueConfigs[-2]
        
        # Vector of n_i
        n_vec = uniqueConfigs[-1]
        
        # multiply with our parameter
        n_vec = k*n_vec
        
        # Vector of priorities
        p_vec = 1 - np.exp(np.divide(-n_vec, n_T))
            
        # Vector of priority probability
        Pr_vec = np.divide(p_vec, np.sum(p_vec))
        
        # Reconstruct a vector that corresponds to each snippet and its priority probability
        Pr_vec_out = Pr_vec[reconst]

        return Pr_vec_out

""" TESTING """

chooseWl = [16]                 # Workloads to be chosen, "This should be a list from" --> [1:2, 5:16, 21, 22, 24, 25]
coreConfig = [1, 1]             # Set the number of little and big cores in the dataset
                                # A15_freq_table = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]                         # A15 frequency table
                                # num_freq = len(A15_freq_table)
TEST_OP = 1                              
env = WorkloadEnv(chooseWl, coreConfig)

df0 = pd.read_csv('data/sorted_all_data.csv', header=None) 
df = env.normalizeHW(df0)
# Get the new data frame and the total number of unique snippets in the data set:
#df2 has [36704 rows x 42 col] for Wl16 and 248 snippets.
df1, numsnippets = env.f_setWorkload(df)

#df2 has [1984 rows x 42 col] for Wl16. In this way, each snippets only have 8 rows
df2 = env.f_setCoreConfig(df1)

#df3 has [1984 rows x 12 col] for Wl16. In this way, each snippets only have 8 rows
df3 = env.f_createFeatureSet(df2)

if TEST_OP == 0:
    # TEST for dataframes ---- DONE, CORRECT
    print(df2)
    print(df3)
    
elif TEST_OP == 1:
    # TEST for Oracle ---- DONE, CORRECT
    oracle_trace = env.oracleTrace(248, df2)
    plt.plot(oracle_trace[:, env.col["freq"]], 'r^-', label="Oracle")
    plt.ylabel('CPU Frequency (GHz)')
    plt.xlabel('Snippets')
    plt.legend()
    plt.show()
elif TEST_OP == 2: 
    # TEST for Next State ---- DONE, CORRECT

    next_state = env.nxtFreq_nxtState(df3, 2.0, 1, 8, 12)
    print(next_state)
    print(df3.loc[15,:])

    next_state = env.nxtFreq_nxtState(df3, 0.6, 3, 8, 12)
    print(next_state)
    print(df3.loc[24,:])
elif TEST_OP == 3: 
    # TEST for next_state, reward, done ---- DONE, CORRECT
    next_state, reward, done = env.f_env(df2, df3, 0, 2.0, 1, 248, 8, 12)
    print(next_state)
    print(reward)
    print(done)

    next_state, reward, done = env.f_env(df2, df3, 0, 1.8, 2, 248, 8, 12)
    print(next_state)
    print(reward)
    print(done)
    
def f_reward(self, dataframe, action, curr_freq, curr_snippet):

    ind_curr_snippet = dataframe.loc[:, self.col["snippet_name"]] == curr_snippet
    candidate_obj = dataframe.loc[ind_curr_snippet, self.col["PPW"]]

    # Find the index of the best PPW for the current snippet
    ind_best_obj = candidate_obj.idxmax()
    optimal_freq = dataframe.loc[ind_best_obj, self.col["freq"]]

    # Check if the current frequency index is same as the optimal index or not
    if (optimal_freq < curr_freq):
        optimal_action = 0
    elif (optimal_freq > curr_freq):
        optimal_action = 2
    else:
        optimal_action = 1
    
    reward = 1 if (action == optimal_action) else -1
    
    return reward, optimal_action, optimal_freq    


