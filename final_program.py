import json
import time
import math
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pyniryo import NiryoRobot, PoseObject, JointsPosition, PinID, PinState
from ultralytics import YOLO
import pyrealsense2 as rs
import threading
from scipy.spatial.transform import Rotation as R

#Configuration Constants
WORKSPACE_VIZ_BOUNDS = {
    "x_min": -0.1, "x_max": 0.2,
    "y_min": -0.4, "y_max": -0.15
}

#General Configuration
GRID_CORNERS = {
    "top_left":     np.array([0.1380, -0.4125]),
    "top_right":    np.array([-0.0329, -0.4125]),
    "bottom_left":  np.array([0.1256, -0.2122]),
    "bottom_right": np.array([-0.0, -0.1987])
}
GRID_ROWS = 4
GRID_COLS = 4
PICK_HOVER_HEIGHT = 0.15
PICK_TARGET_HEIGHT = 0.085
TABLE_Z = 0.088
BLOCK_HEIGHT = 0.025
INITIAL_SCAN_POSE = JointsPosition(0.075, 0.172, -0.195, -0.041, -1.611, 0.081)
PRESS_BUTTON_HOVER_POSE = PoseObject(x=-0.1798, y=0.2466, z=0.209, roll=1.980, pitch=1.485, yaw=-1.607)
PRESS_BUTTON_TOUCH_POSE = PoseObject(x=-0.1798, y=0.2466, z=0.175, roll=1.980, pitch=1.485, yaw=-1.607)

#Pose for final depth verification
DEPTH_DETECTION_POSE = PoseObject(x=0.0387, y=0.1662, z=0.4877, roll=0.2336, pitch=1.3777, yaw=1.5736) #for visualising all workspace
HEIGHT_TOLERANCE = 0.015  # 1.5 cm margin of error

FLANGE_TO_CAM_TRANSFORM = np.array([
    [-0.192962029, 0.265886842, 0.944494491, -0.0644849896],
    [-0.0165481335, 0.961566339, -0.274073593, -0.00642201071],
    [-0.981066672, -0.0685154176, -0.181145861, 0.0167684892],
    [0, 0, 0, 1]
])
WORKSPACE_LIMITS = {
    "x_min": -0.20, "x_max": 0.20,
    "y_min": 0.05,  "y_max": 0.35
}
X_OFFSET = -0.008
Y_OFFSET = -0.010
BASE_PICK_ORIENTATION = {"pitch": 1.550, "yaw": 1.476}

#Main Application Class
class TowerBuilderApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Interactive Multi-Mode Tower Builder")
        self.geometry("1100x800")
        self.model = model
        self.robot = None
        self.pipeline = None
        self.align = None
        self.stop_threads = False
        self.last_button_state = None
        
        self.grid_map_organised = self.load_json_file("organised_grid_map.json")
        self.grid_map_mixed = self.load_json_file("mixed_grid_map.json")
        self.random_objects_list = self.load_json_file("mixed_object_list.json")
        self.angled_objects_list = self.load_json_file("angled_object_list.json")
        
        self.tower_plan = []
        self.blocks_placed = 0
        self.robot_turns_taken = 0
        self.camera_is_paused = threading.Event()
        self.camera_is_paused.set()

        self.container = ttk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (WelcomeScreen, SetupScreen, MonitoringScreen):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(WelcomeScreen)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.event_generate("<<Show>>")
        frame.tkraise()

    def _on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.stop_threads = True
            time.sleep(0.2)
            if self.robot:
                self.robot.set_learning_mode(False)
                self.robot.close_connection()
            if self.pipeline:
                self.pipeline.stop()
            self.destroy()

    def load_json_file(self, filename):
        try:
            with open(filename, "r") as f: return json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Map file '{filename}' not found. Please create it. Error: {e}")
            self.after(10, self.destroy)
            return None
    
    def save_json_file(self, data, filename):
        try:
            with open(filename, "w") as f: json.dump(data, f, indent=4)
        except Exception as e: print(f"Failed to save map {filename}: {e}")
        
    def start_shared_camera(self):
        if self.pipeline is None:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
                self.align = rs.align(rs.stream.color)
                return True
            except Exception as e:
                messagebox.showerror("Camera Error", f"Could not start RealSense camera: {e}")
                self.stop_threads = True
                return False
        return True

#Screen 1: Welcome
class WelcomeScreen(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text="Interactive Tower Building Game", font=("Helvetica", 24)).pack(pady=50)
        ttk.Button(self, text="Begin Setup", command=lambda: controller.show_frame(SetupScreen)).pack(pady=20, ipady=10)

#Screen 2: Setup
class SetupScreen(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        ttk.Label(self, text="Plan Your Tower", font=("Helvetica", 22)).pack(pady=20)
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        num_frame = ttk.Frame(main_frame)
        num_frame.pack(pady=10)
        ttk.Label(num_frame, text="How many blocks to stack?").pack(side="left", padx=5)
        self.num_blocks_var = tk.IntVar(value=6)
        ttk.Spinbox(num_frame, from_=1, to=8, textvariable=self.num_blocks_var, width=5).pack(side="left")
        
        plan_frame = ttk.Frame(main_frame)
        plan_frame.pack(pady=20, fill="x", expand=True)
        plan_frame.columnconfigure(1, weight=1)
        colors_frame = ttk.Frame(plan_frame)
        colors_frame.grid(row=0, column=0, padx=20, sticky="n")
        ttk.Label(colors_frame, text="Click to Add Color:").pack(anchor="w")
        for color in ["red", "green", "blue", "yellow"]:
            ttk.Button(colors_frame, text=color.capitalize(), command=lambda c=color: self.add_color(c)).pack(fill="x", pady=2)
        button_frame = ttk.Frame(plan_frame)
        button_frame.grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="<- Remove Selected", command=self.remove_color).pack(pady=5)
        tower_plan_frame = ttk.Frame(plan_frame)
        tower_plan_frame.grid(row=0, column=2, padx=20)
        ttk.Label(tower_plan_frame, text="Tower Plan (Bottom to Top)").pack()
        self.plan_listbox = tk.Listbox(tower_plan_frame, height=10)
        self.plan_listbox.pack()

        ttk.Button(main_frame, text="Start Game", command=self.start_game).pack(pady=30, ipady=15)

    def add_color(self, color):
        if self.plan_listbox.size() < self.num_blocks_var.get():
            self.plan_listbox.insert(tk.END, color)
        else:
            messagebox.showwarning("Limit Reached", f"Tower plan is full ({self.num_blocks_var.get()} blocks).")

    def remove_color(self):
        selected = self.plan_listbox.curselection()
        if selected: self.plan_listbox.delete(selected[0])

    def start_game(self):
        plan = list(self.plan_listbox.get(0, tk.END))
        if len(plan) != self.num_blocks_var.get():
            messagebox.showerror("Error", "Plan length does not match the number of blocks selected.")
            return
        if not plan:
            messagebox.showerror("Error", "Tower plan is empty.")
            return
        
        self.controller.tower_plan = plan
        self.controller.show_frame(MonitoringScreen)

#Screen 3: Monitoring
class MonitoringScreen(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.is_running = False
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10)
        ttk.Label(left_frame, text="Live Camera Feed", font=("Helvetica", 16)).pack(pady=5)
        self.camera_label = ttk.Label(left_frame)
        self.camera_label.pack(pady=5, expand=True)
        self.status_label = ttk.Label(left_frame, text="Status: Idle", anchor="w", relief="sunken", font=("Helvetica", 14, "bold"))
        self.status_label.pack(fill="x", pady=10)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        self.plan_display_label = ttk.Label(right_frame, text="Building:", font=("Helvetica", 14, "bold"))
        self.plan_display_label.pack(pady=5, anchor="w")
        ttk.Label(right_frame, text="Build Progress", font=("Helvetica", 16)).pack(pady=5)
        self.progress_label = ttk.Label(right_frame, text="0 / 0 Blocks")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(right_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
        ttk.Label(right_frame, text="Tower Visualization", font=("Helvetica", 16)).pack(pady=20)
        self.tower_viz_frame = ttk.Frame(right_frame, relief="sunken")
        self.tower_viz_frame.pack(pady=5, expand=False, fill="x")

        self.mode_map_title = ttk.Label(right_frame, text="Current Mode Map:", font=("Helvetica", 16))
        self.mode_map_title.pack(pady=(20, 5))
        
        self.mode_map_canvas = tk.Canvas(right_frame, width=300, height=300, bg="#cccccc", relief="groove")
        self.mode_map_canvas.pack(pady=5)

        self.bind("<<Show>>", self.on_show)

    def on_show(self, event):
        if self.is_running: return
        self.is_running = True
        self.controller.blocks_placed = 0
        self.controller.robot_turns_taken = 0
        plan_text = ", ".join(self.controller.tower_plan)
        self.plan_display_label.config(text=f"Building: {plan_text}")
        self.progress_bar['maximum'] = len(self.controller.tower_plan)
        self.progress_bar['value'] = 0
        self.progress_label.config(text=f"0 / {len(self.controller.tower_plan)} Blocks")
        self.draw_tower_plan()
        
        threading.Thread(target=self.interactive_multi_mode_worker, daemon=True).start()

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def draw_tower_plan(self):
        for widget in self.tower_viz_frame.winfo_children(): widget.destroy()
        for color in reversed(self.controller.tower_plan):
            tk.Label(self.tower_viz_frame, text=color.upper(), bg="grey", fg="white", font=("Helvetica", 12, "bold"), relief="raised", borderwidth=2).pack(fill="x", ipady=4, padx=20, pady=1)

    def update_tower_viz(self):
        color_map = {'red': '#ff4d4d', 'green': '#4dff4d', 'blue': '#4d4dff', 'yellow': '#ffff4d'}
        children = self.tower_viz_frame.winfo_children()
        for i in range(self.controller.blocks_placed):
            label = children[-(i+1)]
            color = self.controller.tower_plan[i]
            label.config(bg=color_map.get(color, "grey"), fg="black")

    def update_mode_map_display(self, mode_name, map_data):
        self.mode_map_title.config(text=f"Mode Map: {mode_name}")
        self.mode_map_canvas.delete("all")
        color_map = {'red': '#ff4d4d', 'green': '#4dff4d', 'blue': '#4d4dff', 'yellow': '#ffff4d'}
        
        canvas_w = self.mode_map_canvas.winfo_width()
        canvas_h = self.mode_map_canvas.winfo_height()

        if mode_name in ["Organised Rows", "Mixed Rows"]:
            cell_w, cell_h = canvas_w / GRID_COLS, canvas_h / GRID_ROWS
            for r, row_data in enumerate(map_data):
                for c, cell_content in enumerate(row_data):
                    x0, y0 = c * cell_w, r * cell_h
                    x1, y1 = x0 + cell_w, y0 + cell_h
                    bg_color = "grey"
                    if isinstance(cell_content, str):
                        for name, hex_color in color_map.items():
                            if name in cell_content.lower():
                                bg_color = hex_color
                                break
                    self.mode_map_canvas.create_rectangle(x0, y0, x1, y1, fill=bg_color, outline="black")
                    self.mode_map_canvas.create_text(x0 + cell_w/2, y0 + cell_h/2, text=cell_content)
        
        else:
            if not map_data:
                self.mode_map_canvas.create_text(canvas_w/2, canvas_h/2, text="No blocks available", font=("Helvetica", 12))
                return
            
            x_range = WORKSPACE_VIZ_BOUNDS["x_max"] - WORKSPACE_VIZ_BOUNDS["x_min"]
            y_range = WORKSPACE_VIZ_BOUNDS["y_max"] - WORKSPACE_VIZ_BOUNDS["y_min"]
            
            for obj in map_data:
                pos = obj.get("position")
                if not pos: continue
                
                norm_x = (pos['x'] - WORKSPACE_VIZ_BOUNDS["x_min"]) / x_range
                norm_y = (pos['y'] - WORKSPACE_VIZ_BOUNDS["y_min"]) / y_range
                
                canvas_x = norm_x * (canvas_w - 40) + 20
                canvas_y = (1 - norm_y) * (canvas_h - 40) + 20
                
                color_name = obj.get("color", "unknown")
                bg_color = "grey"
                for name, hex_color in color_map.items():
                    if name in color_name.lower():
                        bg_color = hex_color
                        break
                
                self.mode_map_canvas.create_rectangle(canvas_x - 15, canvas_y - 15, canvas_x + 15, canvas_y + 15, fill=bg_color, outline="black")
                self.mode_map_canvas.create_text(canvas_x, canvas_y, text=color_name)

    def camera_worker(self):
        while not self.controller.stop_threads:
            try:
                self.controller.camera_is_paused.wait()
                if self.controller.pipeline is None: continue
                frames = self.controller.pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                img = cv2.resize(np.asanyarray(color_frame.get_data()), (640, 480))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                def update_img():
                    self.camera_label.config(image=img_tk)
                    self.camera_label.image = img_tk
                self.controller.after(0, update_img)
                time.sleep(0.03)
            except Exception as e: print(f"Camera error: {e}"); break

    def interactive_multi_mode_worker(self):
        self.controller.after(0, self.update_status, "Connecting to robot...")
        if not self.controller.start_shared_camera(): return
        threading.Thread(target=self.camera_worker, daemon=True).start()
        
        try:
            robot = NiryoRobot("10.10.10.10")
            self.controller.robot = robot
            robot.calibrate_auto()
            robot.clear_collision_detected()
        except Exception as e:
            self.controller.after(0, self.update_status, f"ERROR: Could not connect to robot: {e}")
            return
            
        picking_modes = [ pick_from_organised_rows, pick_from_mixed_rows, pick_from_random_space, pick_from_angled_pieces ]
        mode_names = ["Organised Rows", "Mixed Rows", "Random Space", "Angled Pieces"]
        map_sources = [ self.controller.grid_map_organised, self.controller.grid_map_mixed, self.controller.random_objects_list, self.controller.angled_objects_list ]

        while self.controller.blocks_placed < len(self.controller.tower_plan) and not self.controller.stop_threads:
            # Human's turn
            next_block_color = self.controller.tower_plan[self.controller.blocks_placed]
            self.controller.after(0, self.update_status, f"Your turn. Place a '{next_block_color}' block and press Green Button.")
            
            while not self.controller.stop_threads:
                try:
                    state = self.controller.robot.digital_read(PinID.DI1)
                    if state != self.controller.last_button_state and state == PinState.HIGH:
                        self.controller.last_button_state = state
                        break
                    self.controller.last_button_state = state
                    time.sleep(0.1)
                except: break
            if self.controller.stop_threads: break
            
            self.controller.blocks_placed += 1
            self.controller.after(0, self.update_tower_viz)
            self.controller.after(0, self.progress_bar.config, {'value': self.controller.blocks_placed})
            self.controller.after(0, self.progress_label.config, {'text': f"{self.controller.blocks_placed} / {len(self.controller.tower_plan)} Blocks"})

            if self.controller.blocks_placed >= len(self.controller.tower_plan):
                self.controller.after(0, self.update_status, "Final block placed by human. Tower complete!")
                break
            
            # Robot's turn
            self.controller.after(0, self.update_status, "Human placed a block. Robot's turn...")
            self.controller.camera_is_paused.clear()
            time.sleep(0.5)
            self.controller.robot.move(INITIAL_SCAN_POSE)
            time.sleep(2)
            stack_pos = find_stack_location(self.controller.robot, self.controller.model, self.controller.pipeline, self.controller.align)
            
            if stack_pos is None:
                self.controller.after(0, self.update_status, "Could not find stack. Aborting.")
                self.controller.camera_is_paused.set()
                break

            mode_index = self.controller.robot_turns_taken % len(picking_modes)
            current_mode_function = picking_modes[mode_index]
            current_mode_name = mode_names[mode_index]
            current_map_data = map_sources[mode_index]
            
            self.controller.after(0, self.update_mode_map_display, current_mode_name, current_map_data)

            robot_block_color = self.controller.tower_plan[self.controller.blocks_placed]
            self.controller.after(0, self.update_status, f"Mode: '{current_mode_name}'. Picking a '{robot_block_color}' block...")
            
            pick_pose = current_mode_function(self.controller, robot_block_color)
            self.controller.camera_is_paused.set()
            
            if pick_pose is None:
                self.controller.after(0, self.update_status, f"ERROR: Could not find '{robot_block_color}' in mode '{current_mode_name}'.")
                break

            place_z = TABLE_Z + (self.controller.blocks_placed * BLOCK_HEIGHT)
            place_pose = PoseObject(x=stack_pos[0], y=stack_pos[1], z=place_z, roll=-0.262, pitch=1.523, yaw=1.228)
            success = execute_full_pick_and_place(self.controller.robot, pick_pose, place_pose)

            if success:
                self.controller.blocks_placed += 1
                self.controller.robot_turns_taken += 1
                self.controller.after(0, self.update_tower_viz)
                self.controller.after(0, self.progress_bar.config, {'value': self.controller.blocks_placed})
                self.controller.after(0, self.progress_label.config, {'text': f"{self.controller.blocks_placed} / {len(self.controller.tower_plan)} Blocks"})

                self.controller.after(0, self.update_status, "Robot pressing button to end turn...")
                self.controller.robot.move(PRESS_BUTTON_HOVER_POSE)
                time.sleep(2)
                self.controller.robot.move(PRESS_BUTTON_TOUCH_POSE)
                time.sleep(1)
                self.controller.robot.move(PRESS_BUTTON_HOVER_POSE)
                time.sleep(2)
                self.controller.robot.move(INITIAL_SCAN_POSE)
            else:
                self.controller.after(0, self.update_status, "ERROR: Pick sequence failed.")
                break
        
        if not self.controller.stop_threads:
            self.controller.after(0, self.update_status, "Game Over! Verifying final tower height...")
            self.controller.camera_is_paused.clear()
            
            is_correct, measured_depth, expected_depth = verify_tower_depth(self.controller, len(self.controller.tower_plan))
            
            self.controller.camera_is_paused.set()

            if is_correct:
                final_message = f"Success! Measured depth ({measured_depth:.3f}m) matches expected ({expected_depth:.3f}m)."
            else:
                final_message = f"Mismatch! Measured depth ({measured_depth:.3f}m) vs expected ({expected_depth:.3f}m)."
            self.controller.after(0, self.update_status, f"Game Over: {final_message}")

#Standalone Functions for Picking Modes & Verification
def verify_tower_depth(controller, total_block_count):
    #Measures the final tower depth and compares it with predicted depth
    robot = controller.robot
    pipeline = controller.pipeline
    
    print("[INFO] Verifying final tower depth...")
    robot.move_pose(DEPTH_DETECTION_POSE)
    time.sleep(2)

    try:
        # Get a stable depth reading from the center of the camera view
        measured_depth = 0
        for _ in range(15):
            frames = pipeline.wait_for_frames()
            aligned_frames = controller.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame: continue
            measured_depth = depth_frame.get_distance(depth_frame.get_width() // 2, depth_frame.get_height() // 2)
            if measured_depth > 0:
                break
        
        if measured_depth == 0:
            print("[ERROR] No valid depth data at the center of the image.")
            return False, 0, 0

        expected_depth = DEPTH_DETECTION_POSE.z - (total_block_count * BLOCK_HEIGHT)

        if abs(measured_depth - expected_depth) <= HEIGHT_TOLERANCE:
            print(f"[SUCCESS] Depth matches. Measured: {measured_depth:.3f}m, Expected: {expected_depth:.3f}m")
            return (True, measured_depth, expected_depth)
        else:
            print(f"[FAILURE] Depth mismatch. Measured: {measured_depth:.3f}m, Expected: {expected_depth:.3f}m")
            return (False, measured_depth, expected_depth)

    except Exception as e:
        print(f"An error occurred during depth verification: {e}")
        return False, 0, 0

def pick_from_organised_rows(controller, color):
    print(f"--- MODE: ORGANISED --- Looking for {color}")
    grid_map = controller.grid_map_organised
    filename = "organised_grid_map.json"
    
    target_loc = None
    for r, row in enumerate(grid_map):
        for c, item in enumerate(row):
            if isinstance(item, str) and color in item.lower():
                target_loc = (r, c)
                break
        if target_loc: break

    if target_loc:
        r, c = target_loc
        pick_x, pick_y = calculate_coords_with_interpolation(r, c)
        grid_map[r][c] = "Empty"
        controller.save_json_file(grid_map, filename)
        return PoseObject(x=pick_x, y=pick_y, z=PICK_TARGET_HEIGHT, roll=3.069, **BASE_PICK_ORIENTATION)
    return None

def pick_from_mixed_rows(controller, color):
    print(f"--- MODE: MIXED --- Looking for {color}")
    grid_map = controller.grid_map_mixed
    filename = "mixed_grid_map.json"
    
    target_loc = None
    for r, row in enumerate(grid_map):
        for c, item in enumerate(row):
            if isinstance(item, str) and color in item.lower():
                target_loc = (r, c)
                break
        if target_loc: break

    if target_loc:
        r, c = target_loc
        pick_x, pick_y = calculate_coords_with_interpolation(r, c)
        grid_map[r][c] = "Empty"
        controller.save_json_file(grid_map, filename)
        return PoseObject(x=pick_x, y=pick_y, z=PICK_TARGET_HEIGHT, roll=3.069, **BASE_PICK_ORIENTATION)
    return None

def pick_from_random_space(controller, color):
    #Finds a block from a list of objects with pre-scanned positions.
    print(f"--- MODE: RANDOM --- Looking for {color}")
    object_list = controller.random_objects_list
    filename = "mixed_object_list.json"
    
    target_obj = None
    for obj in object_list:
        if obj.get("color", "").lower() == color:
            target_obj = obj
            break
            
    if target_obj:
        position = target_obj['position']
        pick_x, pick_y, pick_z = position['x'], position['y'], position['z']
        
        remaining_objects = [obj for obj in object_list if obj is not target_obj]
        controller.random_objects_list = remaining_objects
        controller.save_json_file(remaining_objects, filename)
        
        return PoseObject(x=pick_x, y=pick_y, z=pick_z, roll=3.069, **BASE_PICK_ORIENTATION)
    return None

def pick_from_angled_pieces(controller, color):
    #Finds a block from a list, moves over it, and dynamically detects its angle.
    print(f"--- MODE: ANGLED --- Looking for {color}")
    
    object_list = controller.angled_objects_list
    filename = "angled_object_list.json"
    target_obj = None
    for obj in object_list:
        if obj.get("color", "").lower() == color:
            target_obj = obj
            break
            
    if not target_obj:
        print(f"[ERROR] Could not find color '{color}' in {filename}")
        return None
        
    position = target_obj['position']
    target_x, target_y, pick_z = position['x'], position['y'], position['z']
    
    robot = controller.robot
    hover_pose = PoseObject(x=target_x, y=target_y, z=PICK_HOVER_HEIGHT, roll=3.069, **BASE_PICK_ORIENTATION)
    robot.move_pose(hover_pose)
    time.sleep(2)

    try:
        for _ in range(15):
            frames = controller.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[WARNING] Could not get camera frame for angle detection.")
            return None
        image = np.asanyarray(color_frame.get_data())
        detected_angle_deg = get_block_angle_at_center(image)
    except Exception as e:
        print(f"[ERROR] Vision processing failed: {e}")
        return None

    if detected_angle_deg is None:
        print("[WARNING] Could not detect angle, using neutral grip.")
        detected_angle_deg = 0.0
        
    print(f"-> Detected angle: {detected_angle_deg:.2f} degrees.")
    target_roll_rad = math.radians(detected_angle_deg)

    remaining_objects = [obj for obj in object_list if obj is not target_obj]
    controller.angled_objects_list = remaining_objects
    controller.save_json_file(remaining_objects, filename)
    
    return PoseObject(x=target_x, y=target_y, z=pick_z, roll=target_roll_rad, **BASE_PICK_ORIENTATION)

def get_block_angle_at_center(image):
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not contours: return None
    
    closest_contour = min(contours, key=lambda c: cv2.norm(cv2.minEnclosingCircle(c)[0] - image_center))
    rect = cv2.minAreaRect(closest_contour)
    angle = rect[2]
    width, height = rect[1]
    
    if width < height:
        angle += 90
    if angle > 90:
        angle -= 180
        
    return float(angle)

def execute_full_pick_and_place(robot, pick_pose, place_pose):
    try:
        hover_pick = PoseObject(
            x=pick_pose.x, y=pick_pose.y, z=PICK_HOVER_HEIGHT,
            roll=pick_pose.roll, pitch=pick_pose.pitch, yaw=pick_pose.yaw
        )
        
        robot.move_pose(hover_pick)
        robot.open_gripper()
        robot.move_pose(pick_pose)
        robot.close_gripper()
        time.sleep(1)
        robot.move_pose(hover_pick)
        
        hover_place = PoseObject(
            x=place_pose.x, y=place_pose.y, z=place_pose.z + 0.05,
            roll=place_pose.roll, pitch=place_pose.pitch, yaw=place_pose.yaw
        )

        robot.move_pose(hover_place)
        robot.move_pose(place_pose)
        robot.open_gripper()
        time.sleep(1)
        robot.move_pose(hover_place)
        robot.close_gripper()
        return True
    except Exception as e:
        print(f"Pick/place sequence failed: {e}")
        robot.set_learning_mode(True)
        return False

def calculate_coords_with_interpolation(row, col):
    u = col / (GRID_COLS - 1)
    v = row / (GRID_ROWS - 1)
    p_tl, p_tr, p_bl, p_br = GRID_CORNERS["top_left"], GRID_CORNERS["top_right"], GRID_CORNERS["bottom_left"], GRID_CORNERS["bottom_right"]
    top = p_tl + u * (p_tr - p_tl)
    bot = p_bl + u * (p_br - p_bl)
    final = top + v * (bot - top)
    return final[0], final[1]

def find_stack_location(robot, model, pipeline, align):
    if not pipeline: return None
    try:
        for angle_deg in range(0, 360, 30):
            current_joints = robot.get_joints()
            new_joints = list(current_joints)
            new_joints[0] = math.radians(angle_deg)
            robot.move(JointsPosition(*new_joints))
            time.sleep(2.5)
            
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue
            
            color_image = np.asanyarray(color_frame.get_data())
            results = model(color_image, conf=0.25, verbose=False)
            if results and results[0].boxes:
                best = results[0].boxes[results[0].boxes.conf.argmax()]
                cx, cy = int((best.xyxy[0][0]+best.xyxy[0][2])/2), int((best.xyxy[0][1]+best.xyxy[0][3])/2)
                depth = depth_frame.get_distance(cx, cy)
                if depth > 0:
                    point_cam = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, [cx, cy], depth)
                    T_base_flange = pose_to_matrix(robot.get_pose())
                    T_base_cam = T_base_flange @ FLANGE_TO_CAM_TRANSFORM
                    x_base, y_base, z_base = transform_point(point_cam, T_base_cam)
                    
                    x_detected = x_base + X_OFFSET
                    y_detected = y_base + Y_OFFSET
                    
                    if (WORKSPACE_LIMITS["x_min"] <= x_detected <= WORKSPACE_LIMITS["x_max"] and
                        WORKSPACE_LIMITS["y_min"] <= y_detected <= WORKSPACE_LIMITS["y_max"]):
                        return np.array([x_detected, y_detected, z_base])          
    except Exception as e:
        print(f"Error during stack scan: {e}")
    return None

def pose_to_matrix(pose):
    r = R.from_euler('xyz', [pose.roll, pose.pitch, pose.yaw])
    m = np.identity(4)
    m[:3, :3] = r.as_matrix()
    m[:3, 3] = [pose.x, pose.y, pose.z]
    return m

def transform_point(point_in_camera, transform_matrix):
    p_hom = np.append(point_in_camera, 1)
    p_base = transform_matrix @ p_hom
    return p_base[:3]
    
if __name__ == "__main__":
    try:
        model = YOLO("model_4_blocks.pt")
        app = TowerBuilderApp(model)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"A critical error occurred on startup:\n{e}")

































