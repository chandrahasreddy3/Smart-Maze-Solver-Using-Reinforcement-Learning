import json
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pygame
import time
import os
import cv2
from QLearning import MazeEnv

# Function to select file using OS GUI
def select_maze_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Maze JSON File", filetypes=[("JSON files", "*.json")])
    return file_path


def draw_legend(screen, font, screen_width, cell_size):
    legend_items = [
        ("Start", (0, 100, 0)),  # Dark Green
        ("End", (255, 0, 0)),
        ("Path", (255, 255, 255)),
        ("Wall", (0, 0, 0)),
        ("Agent", (255, 165, 0)),
        ("Progress Bar", (0, 255, 0))
    ]

    items_per_row = 3
    item_width = 200
    legend_height = (len(legend_items) // items_per_row + (len(legend_items) % items_per_row > 0)) * 50 + 60

    legend_surface = pygame.Surface((screen_width, legend_height))
    legend_surface.fill((50, 50, 50))

    total_legend_width = items_per_row * item_width
    legend_x = (screen_width - total_legend_width) // 2
    y = 60

    for index, (text, color) in enumerate(legend_items):
        if index % items_per_row == 0 and index != 0:
            y += 50
            legend_x = (screen_width - total_legend_width) // 2

        pygame.draw.rect(legend_surface, color, (legend_x, y + 5, 20, 20))
        label = font.render(text, True, (255, 255, 255))
        legend_surface.blit(label, (legend_x + 30, y + 5))
        legend_x += item_width

    title_font = pygame.font.SysFont(None, 50, bold=True)
    title_text = title_font.render("Training Visualization....", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(screen_width // 2, 25))
    legend_surface.blit(title_text, title_rect)

    screen.blit(legend_surface, (0, 0))
    return legend_height


def train_q_learning(env, json_file, episodes=5000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, skip_visualization=10, fps=20):
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Suppress Pygame window
    pygame.init()
    pygame.display.set_caption("Training Visualization")
    cell_size = 40
    screen_width = 20 * cell_size
    legend_height = draw_legend(pygame.Surface((screen_width, 1)), pygame.font.SysFont(None, 30), screen_width, cell_size)
    maze_height = env.rows * cell_size + 20
    progress_bar_height = 60
    remaining_space = screen_width - (legend_height + maze_height + progress_bar_height)
    equal_padding = remaining_space // 3

    screen_height = legend_height + maze_height + progress_bar_height + 2 * equal_padding

    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    q_table = np.zeros((env.rows, env.cols, env.action_space.n))

    # Create Folder
    if not os.path.exists("Training Videos"):
        os.makedirs("Training Videos")

    json_name = os.path.basename(json_file).split('.')[0]
    video_path = os.path.join("Training Videos", f"{json_name}.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (screen_width, screen_height))

    font = pygame.font.SysFont(None, 30)
    optimal_path = []

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}", end="\r")  # Display progress in console

        state = env.start_pos
        env.reset()
        done = False
        visualize = (episode % skip_visualization == 0)
        path = []

        while not done:
            if visualize:
                screen.fill((0, 0, 0))

                # Draw Legend
                legend_height = draw_legend(screen, font, screen_width, cell_size)

                # Draw Maze
                padding = (screen_width - env.cols * cell_size) // 2
                maze_y = legend_height + equal_padding
                for row in range(env.rows):
                    for col in range(env.cols):
                        color = (255, 255, 255) if env.maze[row, col] == 1 else (0, 0, 0)
                        pygame.draw.rect(screen, color, (col * cell_size + padding, row * cell_size + maze_y, cell_size, cell_size))
                        if (row, col) == env.start_pos:
                            pygame.draw.rect(screen, (0, 100, 0), (col * cell_size + padding, row * cell_size + maze_y, cell_size, cell_size))
                        if (row, col) == env.end_pos:
                            pygame.draw.rect(screen, (255, 0, 0), (col * cell_size + padding, row * cell_size + maze_y, cell_size, cell_size))

                # Draw Agent
                pygame.draw.rect(screen, (255, 165, 0), (state[1] * cell_size + padding, state[0] * cell_size + maze_y, cell_size, cell_size))

                # Draw Progress Bar at the Bottom
                progress_bar_y = screen_height - progress_bar_height  # Align to the bottom
                pygame.draw.rect(screen, (50, 50, 50), (0, progress_bar_y, screen_width, progress_bar_height))  # Background bar
                progress_percentage = (episode / episodes) * 100
                pygame.draw.rect(screen, (0, 255, 0), (0, progress_bar_y, int(screen_width * (episode / episodes)), progress_bar_height))  # Progress bar
                progress_text = font.render(f"Training Progress: {int(progress_percentage)}%", True, (255, 255, 255))
                screen.blit(progress_text, (screen_width // 2 - 100, progress_bar_y + 15))

                pygame.image.save(screen, "frame.png")
                frame = cv2.imread("frame.png")
                video_writer.write(frame)
                clock.tick(fps)

            # Action Selection
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_table[state[0], state[1]])

            path.append(state)
            next_state, reward, done, _ = env.step(action)
            next_state_tuple = tuple(env.agent_pos)

            q_table[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(q_table[next_state_tuple[0], next_state_tuple[1]]) - q_table[state[0], state[1], action]
            )

            state = next_state_tuple

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    video_writer.release()
                    pygame.quit()
                    return

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode == episodes - 1:
            optimal_path = path

    pygame.quit()
    video_writer.release()
    os.remove("frame.png")

    print(f"\n🎥 Training Video Saved at: {video_path}")


if __name__ == "__main__":
    maze_file = select_maze_file()
    if maze_file:
        env = MazeEnv(maze_file)
        env.reset()
        env.render()

    train_q_learning(env, maze_file, episodes=200, skip_visualization=10, fps=120)
