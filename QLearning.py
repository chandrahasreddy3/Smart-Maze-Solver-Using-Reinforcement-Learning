import json
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pygame
import time

class MazeEnv(gym.Env):
    def __init__(self, maze_file):
        super(MazeEnv, self).__init__()
        
        # Load maze from JSON file
        with open(maze_file, "r") as file:
            data = json.load(file)
        
        self.maze = np.array(data["maze"], dtype=np.int32)
        self.start_pos = tuple(data["start"])
        self.end_pos = tuple(data["end"])
        
        self.rows, self.cols = self.maze.shape
        self.agent_pos = self.start_pos
        
        # Define action space: 0 = Up, 1 = Down, 2 = Left, 3 = Right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space as a grid of 0 (free) and 1 (wall)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=np.int32)

    def reset(self):
        """Resets the environment to the initial state."""
        self.agent_pos = self.start_pos
        return self.get_observation()
    
    def get_observation(self):
        """Returns the current state of the maze with the agent's position."""
        obs = np.copy(self.maze)
        obs[self.agent_pos] = 2  # Represent agent as '2' in the matrix
        return obs
    
    def is_path_left(self, state, direction):
        """Check if there's a path on the left relative to the bot's direction."""
        x, y = state
        if direction == "Up":
            return self.maze[x, y - 1] == 1 if y > 0 else False
        elif direction == "Down":
            return self.maze[x, y + 1] == 1 if y < self.cols - 1 else False
        elif direction == "Left":
            return self.maze[x + 1, y] == 1 if x < self.rows - 1 else False
        elif direction == "Right":
            return self.maze[x - 1, y] == 1 if x > 0 else False
        return False

    def is_path_right(self, state, direction):
        """Check if there's a path on the right relative to the bot's direction."""
        x, y = state
        if direction == "Up":
            return self.maze[x, y + 1] == 1 if y < self.cols - 1 else False
        elif direction == "Down":
            return self.maze[x, y - 1] == 1 if y > 0 else False
        elif direction == "Left":
            return self.maze[x - 1, y] == 1 if x > 0 else False
        elif direction == "Right":
            return self.maze[x + 1, y] == 1 if x < self.rows - 1 else False
        return False
    
    def step(self, action):
        """Takes an action and updates the environment."""
        row, col = self.agent_pos
        
        if action == 0 and row > 0:  # Up
            new_pos = (row - 1, col)
        elif action == 1 and row < self.rows - 1:  # Down
            new_pos = (row + 1, col)
        elif action == 2 and col > 0:  # Left
            new_pos = (row, col - 1)
        elif action == 3 and col < self.cols - 1:  # Right
            new_pos = (row, col + 1)
        else:
            new_pos = self.agent_pos  # Invalid move, stay in place
        
        # Check if the new position is a wall
        if self.maze[new_pos] == 0:  # Now 0 means wall
            new_pos = self.agent_pos  # Stay in place if hitting a wall
        
        self.agent_pos = new_pos
        
        # Define reward structure
        if self.agent_pos == self.end_pos:
            reward = 10  # Reward for reaching the goal
            done = True
        else:
            reward = -0.1  # Small penalty for each step
            done = False
        
        return self.get_observation(), reward, done, {}
    
    def render(self):
        """Displays the maze with black (walls), white (paths), green (start), and red (end) colors."""
        maze_copy = np.copy(self.maze)
        
        # Create RGB image
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        img[maze_copy == 0] = [0, 0, 0]  # Black for walls
        img[maze_copy == 1] = [255, 255, 255]  # White for paths
        img[self.start_pos] = [0, 255, 0]  # Green for start
        img[self.end_pos] = [255, 0, 0]  # Red for end
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    
    def close(self):
        pass

# Function to select file using OS GUI
def select_maze_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Maze JSON File", filetypes=[("JSON files", "*.json")])
    return file_path

# Q-Learning Training Code
def train_q_learning(env, json_file, episodes=5000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    q_table = np.zeros((env.rows, env.cols, env.action_space.n))
    optimal_path = []  # To store the best path

    for episode in range(episodes):
        state = env.start_pos
        env.reset()
        done = False
        episode_path = []  # Track path in this episode
        
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)  # Explore
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit
            
            next_state, reward, done, _ = env.step(action)
            next_state_tuple = tuple(env.agent_pos)
            
            # Q-learning update
            q_table[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(q_table[next_state_tuple[0], next_state_tuple[1]]) - q_table[state[0], state[1], action]
            )

            # Save step in episode path
            episode_path.append((state, action))  
            state = next_state_tuple
        
            # If goal is reached, include the final state in the path
            if done and state == env.end_pos:
                episode_path.append((state, None))  # Append final state with None action
            
        # Save path if this was a successful episode (reaching the goal)
        if state == env.end_pos:
            optimal_path = episode_path  # Save the final successful episode path

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Save Q-table and optimal path
    np.save("q_table.npy", q_table)
    with open(json_file, "r+") as f:
        data = json.load(f)
        data["optimal_path"] = [(list(map(int, state)), int(action) if action is not None else None) for state, action in optimal_path]
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    return q_table, optimal_path




# Visualization using pygame
def visualize_q_learning_pygame(env, q_table):
    pygame.init()
    cell_size = 40
    screen = pygame.display.set_mode((env.cols * cell_size, env.rows * cell_size))
    clock = pygame.time.Clock()
    
    state = env.start_pos
    env.reset()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        screen.fill((0, 0, 0))
        for row in range(env.rows):
            for col in range(env.cols):
                color = (255, 255, 255) if env.maze[row, col] == 1 else (0, 0, 0)
                pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (0, 255, 0) if (row, col) == env.start_pos else (255, 0, 0) if (row, col) == env.end_pos else color, (col * cell_size, row * cell_size, cell_size, cell_size))
        
        pygame.draw.rect(screen, (255, 165, 0), (state[1] * cell_size, state[0] * cell_size, cell_size, cell_size))
        pygame.display.flip()
        
        action = np.argmax(q_table[state[0], state[1]])
        _, _, done, _ = env.step(action)
        state = tuple(env.agent_pos)
        clock.tick(5)
    
    pygame.quit()



if __name__ == "__main__":
    maze_file = select_maze_file()
    if maze_file:
        env = MazeEnv(maze_file)
        env.reset()
        env.render()

    q_table, optimal_path = train_q_learning(env, maze_file)
    # print("Training Complete! Q-Table saved as 'q_table.npy'.")

    visualize_q_learning_pygame(env, q_table)
