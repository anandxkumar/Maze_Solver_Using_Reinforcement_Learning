#Creating maze environment
import numpy as np
import time

import tkinter as tk

# Number of units in a single block
UNIT = 40

Maze_Height = 6
Maze_Width = 6

class Maze():
    
    def __init__(self):
        # We need an instance of tk.Tk class. The tk.Tk class is a top-level widget of Tk and serves as the main window of the application.
        #window variable will be the window
        self.window = tk.Tk()
        self.window.title("Maze using Q-Learning")
        # Types of action that can be taken at any state
        self.action_space=['u','d','l','r']
        #Defining Dimensions of the window
        self.window.geometry('{}x{}'.format(Maze_Width*UNIT,Maze_Height*UNIT))
        self.no_action=len(self.action_space)
        self.build_maze()
        
    def build_maze(self):
        self.canvas = tk.Canvas(self.window,bg = 'white',width = Maze_Width*UNIT, height = Maze_Height*UNIT)
        
        #Creating Vertical Lines
        
        for c in range(0, Maze_Width*UNIT,UNIT):
            x0, y0, x1, y1 = c, 0, c, Maze_Height*UNIT
            self.canvas.create_line(x0,y0,x1,y1)
         
        #Creating Horizontal Lines   
         
        for r in range(0, Maze_Height*UNIT,UNIT):
            x0, y0, x1, y1 = 0, r, Maze_Width*UNIT, r
            self.canvas.create_line(x0,y0,x1,y1)
            
        # create origin point ( It is the center of the first cell in the first row)    
        origin = np.array([20,20]) # Initial Point of Player
        
        #Creating Obstacles in the maze
        
        obstacle1_center = origin + np.array([UNIT*2,UNIT])
        self.obstacle1 = self.canvas.create_rectangle(
            obstacle1_center[0]-15,obstacle1_center[1]-15,
            obstacle1_center[0]+15,obstacle1_center[1]+15,fill = 'black'
            )
        
        obstacle2_center = origin + np.array([UNIT,UNIT*2])
        self.obstacle2 = self.canvas.create_rectangle(
            obstacle2_center[0]-15,obstacle2_center[1]-15,
            obstacle2_center[0]+15,obstacle2_center[1]+15,fill = 'black'
            )
        
        obstacle3_center = origin + np.array([UNIT*4,UNIT*4])
        self.obstacle3 = self.canvas.create_rectangle(
            obstacle3_center[0]-15,obstacle3_center[1]-15,
            obstacle3_center[0]+15,obstacle3_center[1]+15,fill = 'black'
            )
        
        obstacle4_center = origin + np.array([UNIT*3,UNIT*2])
        self.obstacle4 = self.canvas.create_rectangle(
            obstacle4_center[0]-15,obstacle4_center[1]-15,
            obstacle4_center[0]+15,obstacle4_center[1]+15,fill = 'black'
            )
        
        # The Goal Where player wants to reach
        
        goal_center = origin + UNIT*2
        
        self.goal = self.canvas.create_oval(
            goal_center[0]-15, goal_center[1]-15,
            goal_center[0]+15, goal_center[1]+15, fill = 'yellow'
            )
        
        # rect means current position
        
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15, fill = 'red'
            )

        # TO add all features in the canvas
        
        self.canvas.pack()
        
    # To render window after an interval   
    def render(self):
        time.sleep(0.1)
        self.window.update()
        
    # To reset the player position
    def reset(self):
        
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        
        
        origin = np.array([20,20])
        
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill = 'red'
            )
        #Returns coordinates of rectangle / observation
        return self.canvas.coords(self.rect)
        
    # Get state and reward
    
    def get_state_reward(self, action):
        # s contains current coordinates
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])
        
        if action == 0 : # up
            # y coordinate of rect    
            if s[1] > UNIT : # Means the rect is below the first row
                # Moving The rect to up by one unit
                base_action[1] -= UNIT
                
        elif action == 1 : #down
            if s[1] < (Maze_Height-1)*UNIT:
                base_action[1] += UNIT
                
        
        elif action == 2 : # right
            if s[0] < (Maze_Width - 1)* UNIT:
                base_action[0] += UNIT
                
        elif action == 3 : # Left
            if s[0] > UNIT :
                base_action[0] -= UNIT
        
        # Moves the rect to new coordinates by adding base_action coordinates to the rect
        self.canvas.move(self.rect,base_action[0],base_action[1])
        # New Coordinates
        s_ = self.canvas.coords(self.rect)
        
        
        # REWARD SYSTEM
        
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            s_ = 'terminal' #We have reached final state
            
        elif s_ in [self.canvas.coords(self.obstacle1) , self.canvas.coords(self.obstacle2), self.canvas.coords(self.obstacle3),
                    self.canvas.coords(self.obstacle4)]:
            reward = -1
            done = True 
            s_ = 'terminal'
            
        else :
            reward = 0
            done = False
            
        return s_, reward, done
            
        
        
if __name__ == '__main__':
    maze = Maze()
    maze.window.mainloop()

        
        