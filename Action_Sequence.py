import json
import numpy as np
import tkinter as tk
from tkinter import messagebox
from QLearning import MazeEnv  # Import your MazeEnv class

def select_maze_file():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Maze JSON File", filetypes=[("JSON files", "*.json")])
    return file_path


class MazeBot:
    def __init__(self, maze_env, optimal_path, start_direction):
        self.maze_env = maze_env  # Custom MazeEnv instance
        self.optimal_path = optimal_path  # List of states from JSON optimal path
        self.direction = start_direction  # Initial direction ('N', 'E', 'S', 'W')
        self.actions = []  # Stores sequence of actions
        self.last_action_was_turn = False  # Track if last action was a turn
    
    def turn_right(self, current_state):
        dir_map = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
        left_path, mid_path, right_path = self.get_paths(current_state)
        self.direction = dir_map[self.direction]
        self.actions.append([left_path, mid_path, right_path, "Right"])
        self.last_action_was_turn = True
    
    def turn_left(self, current_state):
        dir_map = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
        left_path, mid_path, right_path = self.get_paths(current_state)
        self.direction = dir_map[self.direction]
        self.actions.append([left_path, mid_path, right_path, "Left"])
        self.last_action_was_turn = True
    
    def move_forward(self, current_state):
        if self.last_action_was_turn:
            self.last_action_was_turn = False
            return  # Ignore first forward after a turn
        left_path, mid_path, right_path = self.get_paths(current_state)
        self.actions.append([left_path, mid_path, right_path, "Forward"])
    
    def get_paths(self, state):
        x, y = state
        left_path, mid_path, right_path = 0, 1 if self.maze_env.maze[x, y] == 1 else 0, 0
        if self.direction == "N":
            left_path = 1 if y > 0 and self.maze_env.maze[x, y - 1] == 1 else 0
            right_path = 1 if y < self.maze_env.cols - 1 and self.maze_env.maze[x, y + 1] == 1 else 0
        elif self.direction == "S":
            left_path = 1 if y < self.maze_env.cols - 1 and self.maze_env.maze[x, y + 1] == 1 else 0
            right_path = 1 if y > 0 and self.maze_env.maze[x, y - 1] == 1 else 0
        elif self.direction == "E":
            left_path = 1 if x > 0 and self.maze_env.maze[x - 1, y] == 1 else 0
            right_path = 1 if x < self.maze_env.rows - 1 and self.maze_env.maze[x + 1, y] == 1 else 0
        elif self.direction == "W":
            left_path = 1 if x < self.maze_env.rows - 1 and self.maze_env.maze[x + 1, y] == 1 else 0
            right_path = 1 if x > 0 and self.maze_env.maze[x - 1, y] == 1 else 0
        return left_path, mid_path, right_path
    
    def generate_action_sequence(self):
        for i in range(len(self.optimal_path) - 1):
            current_state, _ = self.optimal_path[i]  # Extract (x, y) position
            next_state, _ = self.optimal_path[i + 1]  # Extract (x, y) position
            
            # Determine movement direction
            dx, dy = next_state[0] - current_state[0], next_state[1] - current_state[1]
            movement = None
            if dx == -1:  # Moving up
                movement = 'N'
            elif dx == 1:  # Moving down
                movement = 'S'
            elif dy == -1:  # Moving left
                movement = 'W'
            elif dy == 1:  # Moving right
                movement = 'E'
            
            if movement:
                while self.direction != movement:
                    if (self.direction == 'N' and movement == 'E') or \
                       (self.direction == 'E' and movement == 'S') or \
                       (self.direction == 'S' and movement == 'W') or \
                       (self.direction == 'W' and movement == 'N'):
                        self.turn_right(current_state)
                    else:
                        self.turn_left(current_state)
                self.move_forward(current_state)
        
        # Append final goal state with stop action
        final_state, _ = self.optimal_path[-1]
        left_path, mid_path, right_path = self.get_paths(final_state)
        self.actions.append([left_path, mid_path, right_path, "Stop"])

        # Check previous state paths and append [0, 0, 0, "Stop"] if equal to goal state paths
        if len(self.optimal_path) > 1:
            prev_state, _ = self.optimal_path[-2]
            prev_left, prev_mid, prev_right = self.get_paths(prev_state)
            if (prev_left, prev_mid, prev_right) == (left_path, mid_path, right_path):
                self.actions[-1] = [0, 0, 0, "Stop"]
        
        return self.actions
    

def remove_consecutive_repeats(sequence):
    cleaned_sequence = []
    previous = None

    for item in sequence:
        # Check if the current item is the same as the previous one
        if item != previous:
            cleaned_sequence.append(item)
        previous = item

    return cleaned_sequence


if __name__ == "__main__":
    # Select maze file and initialize environment
    maze_file = select_maze_file()
    if maze_file:
        env = MazeEnv(maze_file)
        env.reset()
        env.render()

    # Load trained optimal path from JSON
    with open(maze_file, "r") as f:
        data = json.load(f)
    
    if "optimal_path" in data:
        bot = MazeBot(env, data["optimal_path"], data["direction"])
        action_sequence = bot.generate_action_sequence()
        cleaned_actions = remove_consecutive_repeats(action_sequence)

        # Print cleaned sequence
        for action in cleaned_actions:
            print(action)

        data["action_sequence"] = cleaned_actions
        with open(maze_file, "w") as f:
            json.dump(data, f)
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", "The Selected maze does not have an optimal path!\nTrain the agent in the maze and Try again!")

