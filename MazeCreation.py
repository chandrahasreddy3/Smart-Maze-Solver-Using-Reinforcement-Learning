import tkinter as tk
from tkinter import messagebox
import numpy as np
import json
import os 

class MazeEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Editor")

        self.rows = 15
        self.cols = 20
        self.cell_size = 25
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.last_selected = None
        self.start_pos = None
        self.end_pos = None
        self.selected_color = "black"
        self.selected_direction = tk.StringVar()
        self.selected_direction.set("North")  # Default direction
        self.history = []

        # Mapping Full Names to Abbreviations
        self.direction_map = {
            "North": "N",
            "South": "S",
            "East": "E",
            "West": "W"
        }

        self.create_controls()
        self.reset_canvas()

    def create_controls(self):
        self.create_color_selector()
        self.create_direction_selector()

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        undo_button = tk.Button(control_frame, text="Undo", command=self.undo, font=("Arial", 10, "bold"), width=10)
        undo_button.pack(side=tk.LEFT, padx=10)

        reset_button = tk.Button(control_frame, text="Reset", command=self.reset_maze, font=("Arial", 10, "bold"), width=10)
        reset_button.pack(side=tk.LEFT, padx=10)

        save_frame = tk.Frame(self.root)
        save_frame.pack(pady=10)

        tk.Label(save_frame, text="Maze Name:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.file_name_entry = tk.Entry(save_frame, width=15)
        self.file_name_entry.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(save_frame, text="Save", command=self.save_maze)
        save_button.pack(side=tk.LEFT, padx=10)

    def reset_canvas(self):
        if hasattr(self, 'canvas'):
            self.canvas.destroy()
        self.create_canvas()

    def create_canvas(self):
        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size

        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.toggle_cell)
        self.canvas.bind("<B1-Motion>", self.toggle_cell)
        self.draw_grid()

    def create_color_selector(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        colors = {"Path": "black", "Start": "green", "End": "red"}
        for label, color in colors.items():
            btn = tk.Button(frame, text=label, bg=color, fg="white", font=("Arial", 10, "bold"), width=10,
                            command=lambda c=color: self.set_selected_color(c))
            btn.pack(side=tk.LEFT, padx=10)

    def create_direction_selector(self):
        direction_frame = tk.Frame(self.root)
        direction_frame.pack(pady=10)
        tk.Label(direction_frame, text="Direction:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        directions = ["North", "South", "East", "West"]
        for direction in directions:
            rb = tk.Radiobutton(direction_frame, text=direction, variable=self.selected_direction,
                                value=direction, font=("Arial", 10))
            rb.pack(side=tk.LEFT, padx=5)

    def set_selected_color(self, color):
        self.selected_color = color

    def draw_grid(self):
        self.canvas.delete("all")
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                if (row, col) == self.start_pos:
                    color = "red"
                elif (row, col) == self.end_pos:
                    color = "green"
                else:
                    color = "black" if self.grid[row, col] == 1 else "white"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def toggle_cell(self, event):
        if self.grid is None:
            return

        col, row = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.last_selected == (row, col):
                return
            self.last_selected = (row, col)
            self.history.append((row, col, self.grid[row, col]))

            if self.selected_color == "black":
                self.grid[row, col] = 1 - self.grid[row, col]
            elif self.selected_color == "red":
                if self.start_pos is not None:
                    messagebox.showwarning("Warning", "Only one start position allowed!")
                    return
                self.start_pos = (row, col)
            elif self.selected_color == "green":
                if self.end_pos is not None:
                    messagebox.showwarning("Warning", "Only one end position allowed!")
                    return
                self.end_pos = (row, col)

            self.draw_grid()

    def undo(self):
        if self.history:
            row, col, prev_value = self.history.pop()
            self.grid[row, col] = prev_value
            self.draw_grid()

    def reset_maze(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.start_pos = None
        self.end_pos = None
        self.history.clear()
        self.draw_grid()

    def save_maze(self):
        if self.grid is None:
            messagebox.showwarning("Warning", "Please apply dimensions before saving the maze.")
            return

        file_name = self.file_name_entry.get().strip()
        if not file_name:
            messagebox.showwarning("Warning", "Please enter a maze name before saving.")
            return

        confirm = messagebox.askyesno("Verify Maze", "Do you want to save this maze?")
        if confirm:
            selected_dir = self.selected_direction.get()
            maze_data = {
                "maze": self.grid.tolist(),
                "start": self.start_pos if self.start_pos else [0, 0],
                "end": self.end_pos if self.end_pos else [self.rows - 1, self.cols - 1],
                "direction": self.direction_map[selected_dir]
            }

            folder_name = "Maze JSON files"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            file_path = os.path.join(folder_name, f"{file_name}.json")
            with open(file_path, "w") as file:
                json.dump(maze_data, file)

            messagebox.showinfo("Success", f"Maze saved as {file_path}")
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeEditor(root)
    root.mainloop()
