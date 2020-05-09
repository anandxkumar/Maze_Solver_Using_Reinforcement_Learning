# Interaction between agent and environment

from maze_env import Maze 
from RL_agent import QlearningTable

import matplotlib

# Backend 
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

episode_count = 100 # number of epochs
episodes = range(episode_count) 
rewards = []  # The gained reward in each episode
movements = []   # Number of movements happened in each episode

def run_exp():
    for episode in episodes:
        print ("Episode {}/{}".format(episode+1, episode_count))
        
        observation = env.reset()
        moves = 0
        
        while True :
            
            env.render()
            
            # Q-learning chooses action based on observation
            # we convert observation to str since we want to use them as index for our DataFrame.
            action = q_learning_agent.choose_action(str(observation))
            
            # RL takes action and gets next observation and reward
            observation_, reward, done = env.get_state_reward(action)
            moves += 1
            
            # RL learn from the above transition,
            # Update the Q value for the given tuple
            
            q_learning_agent.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            
            if done:
                movements.append(moves)
                rewards.append(reward)
                print("Reward : {} , Moves : {}".format(reward,moves))
                break
        
        
    print(" Game Over")
    plot_reward_movements()    
        
def plot_reward_movements():
    plt.figure()
    plt.subplot(2,1,1) # Number of rows, columns, index
    #episodes = np.asarray(episodes)
    #movements = np.asarray(movements)
    plt.plot(episodes, movements)
    plt.xlabel("Episode")
    plt.ylabel("Movements")
    
    plt.subplot(2,1,2)
    plt.step(episodes,rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("reward_movement_qlearning,png")
    plt.show()    
            
            
if __name__ == '__main__':
    env = Maze()
    q_learning_agent = QlearningTable(actions = list(range(env.no_action)))
    # Call run_experiment() function once after given time in milliseconds.
    env.window.after(10, run_exp)
    env.window.mainloop()

