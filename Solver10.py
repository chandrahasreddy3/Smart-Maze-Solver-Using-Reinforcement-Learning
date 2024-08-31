import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from MazeEnvironment import maze2 as VM

class ComplexMazeEnv(gym.Env):
    def __init__(self):
        super(ComplexMazeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=99, shape=(2,), dtype=np.int32)  # Agent's position

        self.state = None
        self.goal = (47, 91)  # Goal position (Green) - Adjusted for 10x10 bot
        self.start = (47, 0)  # Start position (Red) - Adjusted for 10x10 bot
        self.bot_size = 10
        self.step_size = 5  # Step size for movement
        self.maze = VM  # Assuming the maze is loaded from a numpy file

    def reset(self):
        self.state = list(self.start)
        return self.state

    def step(self, action):
        old_state = self.state.copy()

        if action == 0:  # Up
            new_state = [self.state[0], self.state[1] - self.step_size]
        elif action == 1:  # Down
            new_state = [self.state[0], self.state[1] + self.step_size]
        elif action == 2:  # Right
            new_state = [self.state[0] + self.step_size, self.state[1]]
        elif action == 3:  # Left
            new_state = [self.state[0] - self.step_size, self.state[1]]

        # Check if the new state is within bounds and does not collide with walls
        if self.is_valid_position(new_state):
            self.state = new_state
        else:
            # If the new state is invalid, revert to the old state
            self.state = old_state

        if self.is_goal(self.state):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, {}

    def is_valid_position(self, position):
        x, y = position
        if x < 0 or y < 0 or x + self.bot_size > self.maze.shape[1] or y + self.bot_size > self.maze.shape[0]:
            return False
        for i in range(self.bot_size):
            for j in range(self.bot_size):
                if self.maze[y + j, x + i] == 1:
                    return False
        return True

    def is_goal(self, position):
        x, y = position
        goal_x, goal_y = self.goal
        return x <= goal_x < x + self.bot_size and y <= goal_y < y + self.bot_size

    def render(self):
        if self.state is None:
            print("Environment not reset. Call reset() before render().")
            return
        
        maze_copy = np.copy(self.maze)
        # Mark the agent position
        for i in range(self.bot_size):
            for j in range(self.bot_size):
                maze_copy[self.state[1] + j, self.state[0] + i] = 2
        # Mark the goal position
        for i in range(self.bot_size):
            for j in range(self.bot_size):
                maze_copy[self.goal[1] + j, self.goal[0] + i] = 3

        plt.imshow(maze_copy, cmap="hot", interpolation="nearest")
        plt.title('2D Maze Environment')
        plt.colorbar()
        plt.show()

# Initialize the environment
env = ComplexMazeEnv()

# Ensure the environment is reset before rendering
env.reset()
env.render()






import random
import numpy as np

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 5000

# Initialize Q-table for 100x100 maze and 4 actions
q_table = np.zeros((100 // 5, 100 // 5, 4))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state[0], state[1]])  # Exploit

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
    q_table[state[0], state[1], action] += alpha * (td_target - q_table[state[0], state[1], action])

# Training loop
for episode in range(episodes):
    print(f"Episode: {episode}")
    state = env.reset()
    done = False

    while not done:
        action = choose_action((state[0] // env.step_size, state[1] // env.step_size))
        next_state, reward, done, _ = env.step(action)
        update_q_table((state[0] // env.step_size, state[1] // env.step_size), action, reward, (next_state[0] // env.step_size, next_state[1] // env.step_size))
        state = next_state

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training finished.")

# Save the trained Q-table
np.save("q_table.npy", q_table)








import pygame
import numpy as np
import time

# Load the trained Q-table
q_table = np.load("q_table.npy")

# Initialize the environment
env = ComplexMazeEnv()

# Pygame setup
pygame.init()
size = width, height = 500, 500  # Reduced window size
block_size = 5  # Adjusted block size

screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def draw_maze(env):
    for y in range(env.maze.shape[0]):
        for x in range(env.maze.shape[1]):
            color = WHITE if env.maze[y, x] == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(x * block_size, y * block_size, block_size, block_size))

def draw_agent(state, bot_size):
    for i in range(bot_size):
        for j in range(bot_size):
            pygame.draw.rect(screen, RED, pygame.Rect((state[0] + i) * block_size, (state[1] + j) * block_size, block_size, block_size))

def draw_goal(goal, bot_size):
    for i in range(bot_size):
        for j in range(bot_size):
            pygame.draw.rect(screen, GREEN, pygame.Rect((goal[0] + i) * block_size, (goal[1] + j) * block_size, block_size, block_size))

def main():
    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill(BLACK)
        draw_maze(env)
        draw_agent(state, env.bot_size)
        draw_goal(env.goal, env.bot_size)
        pygame.display.flip()

        action = np.argmax(q_table[state[0] // env.step_size, state[1] // env.step_size])
        next_state, reward, done, _ = env.step(action)
        state = next_state

        time.sleep(0.1)
        clock.tick(10)

    print("Reached the goal!")
    time.sleep(2)
    pygame.quit()

if __name__ == "__main__":
    main()
