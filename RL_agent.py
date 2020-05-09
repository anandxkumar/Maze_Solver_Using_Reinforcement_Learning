import numpy as np
import pandas as pd

class QlearningTable():
    def __init__(self,actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.1):
        
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64) # table is formed with columns u,d,l,r
        
      
    # To choose any action at any state
    
    def choose_action(self, observation):
        
        # action selection based on greedy policy
        
        self.add_state(observation)
        if np.random.uniform() < self.epsilon: # Explore
            action = np.random.choice(self.actions)
            
        else :
            # Exploitation
            # choose best action for the given observation
            #       1)find the records of current observation,
            #       2)reindex the result data and
            #       3)return the action with higherst value.
            state_action = self.q_table.loc[observation, :] # observation is the row
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # Reindexing means that all rows are shuffle
            # np.random.permutation gives array of permuted indexes
            
            action = state_action.idxmax() # Return index of first occurrence of maximum over requested axis.
        return action
    
    def learn(self, s, a ,r ,s_):
        # ToDo: add the next observation (s_) to the table
        self.add_state(s_)
        # Getting the best = q value of 's' state and 'a' action
        q_predict = self.q_table.loc[s,a]
        # check if the next state is a terminal state or not and get the expected q value
        if s_ != 'terminal':
            # Bellman's Equation
            # Q'() = r + gamma * [max_a' Q(s',a')]
            q_target = r + self.gamma*self.q_table.loc[s_,:].max()
            
        else :
            # next state is terminal
            q_target = r
        # Q(s, a) = Q(s, a) + learning_rate [r + gamma max_a' Q(s', a') - Q(s, a)]   
        self.q_table.loc[s,a] += self.lr*(q_target - q_predict)
        
        
        
    def add_state(self, state):
        # append new state to q table
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions),
                          index = self.q_table.columns,
                          name = state) # Name of index
                )

