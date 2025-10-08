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

X_OFFSET = -0.13
Y_OFFSET = 0.0
WORKSPACE_LIMITS = {
    "x_min": -0.20, "x_max": 0.20,
    "y_min": 0.05,  "y_max": 0.35
}

#General Configuration
GRID_CORNERS = {
    "top_left":     np.array([0.121, -0.438]), 
    "top_right":    np.array([-0.080, -0.423]), 
    "bottom_left":  np.array([0.134, -0.214]), 
    "bottom_right": np.array([-0.05, -0.195])
}
GRID_ROWS = 4
GRID_COLS = 4
PICK_HOVER_HEIGHT = 0.15
PICK_TARGET_HEIGHT = 0.09
TABLE_Z = 0.088
BLOCK_HEIGHT = 0.025
HEIGHT_TOLERANCE = 0.015
INITIAL_SCAN_POSE = JointsPosition(0.075, 0.172, -0.195, -0.041, -1.611, 0.081)
PRESS_BUTTON_HOVER_POSE = PoseObject(x=-0.159, y=0.246, z=0.209, roll=0.745, pitch=1.543, yaw=-2.826)
PRESS_BUTTON_TOUCH_POSE = PoseObject(x=-0.158, y=0.246, z=0.172, roll=0.988, pitch=1.503, yaw=-2.535)
FLANGE_TO_CAM_TRANSFORM = np.array([
    [-0.192962029, 0.265886842, 0.944494491, -0.0644849896],
    [-0.0165481335, 0.961566339, -0.274073593, -0.00642201071],
    [-0.981066672, -0.0685154176, -0.181145861, 0.0167684892],
    [0, 0, 0, 1]
])

# --- Main Application Class ---
class TowerBuilderApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Interactive Tower Builder")
        self.geometry("1100x800")
        self.model = model
        self.robot = None
        self.pipeline = None
        self.align = None
        self.stop_threads = False
        self.grid_map = self.load_grid_map()
        self.tower_plan = []
        self.blocks_placed = 0
        self.last_button_state = None
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

    def load_grid_map(self, filename="grid_map.json"):
        try:
            with open(filename, "r") as f: return json.load(f)
        except:
            messagebox.showerror("Error", f"Map file '{filename}' not found. Please create one.")
            self.after(10, self.destroy)
            return None

    def save_grid_map(self):
        try:
            with open("grid_map.json", "w") as f: json.dump(self.grid_map, f, indent=4)
        except Exception as e: print(f"Failed to save map: {e}")
        
    def start_shared_camera(self):
        if self.pipeline is None:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
                self.align = rs.align(rs.stream.color)
                print("[INFO] Shared camera pipeline started.")
                return True
            except Exception as e:
                messagebox.showerror("Camera Error", f"Could not start RealSense camera: {e}")
                self.stop_threads = True
                return False
        return True

#Screen 1: Welcome screen
class WelcomeScreen(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text="Welcome to the Interactive Stacking Game", font=("Helvetica", 24)).pack(pady=50)
        ttk.Button(self, text="Press Here to Continue", command=lambda: controller.show_frame(SetupScreen)).pack(pady=20, ipady=10)

#Screen 2: Tower Setup
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
        self.num_blocks_var = tk.IntVar(value=3)
        ttk.Spinbox(num_frame, from_=1, to=10, textvariable=self.num_blocks_var, width=5).pack(side="left")
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
        ttk.Button(main_frame, text="Start Game", command=self.start_game).pack(pady=30, ipady=10)

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

#Screen 3: Monitoring / Game Screen
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
        self.tower_viz_frame.pack(pady=5, expand=True, fill="both")
        
        ttk.Label(right_frame, text="Available Blocks", font=("Helvetica", 16)).pack(pady=(20, 5))
        grid_frame = ttk.Frame(right_frame, relief="groove", padding=5)
        grid_frame.pack(pady=5)
        self.grid_buttons = [[tk.Button(grid_frame, text="E", width=8, height=3, bg="grey", fg="white", state="disabled") for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        for r, row_btns in enumerate(self.grid_buttons):
            for c, btn in enumerate(row_btns):
                btn.grid(row=r, column=c, padx=2, pady=2)
        
        self.bind("<<Show>>", self.on_show)

    def on_show(self, event):
        if self.is_running: return
        self.is_running = True
        plan_text = ", ".join(self.controller.tower_plan)
        self.plan_display_label.config(text=f"Building: {plan_text}")
        self.progress_bar['maximum'] = len(self.controller.tower_plan)
        self.draw_tower_plan()
        self.update_ui_grid()
        threading.Thread(target=self.game_logic_worker, daemon=True).start()

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def draw_tower_plan(self):
        for widget in self.tower_viz_frame.winfo_children(): widget.destroy()
        color_map = {'red': '#ff4d4d', 'green': '#4dff4d', 'blue': '#4d4dff', 'yellow': '#ffff4d'}
        for color in reversed(self.controller.tower_plan):
            tk.Label(self.tower_viz_frame, text=color.upper(), bg="grey", fg="white", font=("Helvetica", 12, "bold"), relief="raised", borderwidth=2).pack(fill="x", ipady=10, padx=20, pady=2)

    def update_tower_viz(self):
        color_map = {'red': '#ff4d4d', 'green': '#4dff4d', 'blue': '#4d4dff', 'yellow': '#ffff4d'}
        children = self.tower_viz_frame.winfo_children()
        for i in range(self.controller.blocks_placed):
            label = children[-(i+1)]
            color = self.controller.tower_plan[i]
            label.config(bg=color_map.get(color, "grey"), fg="black")

    def update_ui_grid(self):
        color_map = {'red': '#ff4d4d', 'green': '#4dff4d', 'blue': '#4d4dff', 'yellow': '#ffff4d'}
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell_content = self.controller.grid_map[r][c]
                color = "grey"
                if isinstance(cell_content, str):
                    for name, hex_color in color_map.items():
                        if name in cell_content.lower():
                            color = hex_color
                            break
                self.grid_buttons[r][c].config(text=cell_content, bg=color)

    def camera_worker(self):
        while not self.controller.stop_threads:
            try:
                self.controller.camera_is_paused.wait()
                if self.controller.pipeline is None:
                    time.sleep(0.1)
                    continue
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

    def game_logic_worker(self):
        if not self.controller.start_shared_camera(): return
        
        threading.Thread(target=self.camera_worker, daemon=True).start()

        self.controller.after(0, self.update_status, "Connecting to robot...")
        try:
            self.controller.robot = NiryoRobot("10.10.10.10")
            self.controller.robot.calibrate_auto()
        except Exception as e:
            self.controller.after(0, self.update_status, f"ERROR: Could not connect to robot: {e}")
            return

        while self.controller.blocks_placed < len(self.controller.tower_plan) and not self.controller.stop_threads:
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

            self.controller.after(0, self.update_status, "Human turn complete. Robot is scanning...")
            self.controller.camera_is_paused.clear()
            time.sleep(0.5)
            self.controller.robot.move(INITIAL_SCAN_POSE)
            time.sleep(2)
            stack_pos = find_stack_location(self.controller.robot, self.controller.model, self.controller.pipeline, self.controller.align)
            self.controller.camera_is_paused.set()

            if stack_pos is None:
                self.controller.after(0, self.update_status, "Could not find stack. Aborting.")
                break

            robot_block_color = self.controller.tower_plan[self.controller.blocks_placed]
            self.controller.after(0, self.update_status, f"Robot is picking a '{robot_block_color}' block...")
            
            target_location = None
            for r, row in enumerate(self.controller.grid_map):
                for c, item in enumerate(row):
                    if isinstance(item, str) and robot_block_color in item.lower():
                        target_location = (r, c)
                        break
                if target_location: break
            
            if not target_location:
                self.controller.after(0, self.update_status, f"No '{robot_block_color}' blocks found. Aborting.")
                break

            row, col = target_location
            pick_x, pick_y = calculate_coords_with_interpolation(row, col)
            place_z = TABLE_Z + (self.controller.blocks_placed * BLOCK_HEIGHT)
            success = execute_pick_and_place(self.controller.robot, pick_x, pick_y, place_z, stack_pos)

            if success:
                self.controller.grid_map[row][col] = "Empty"
                self.controller.save_grid_map()
                self.controller.blocks_placed += 1
                self.controller.after(0, self.update_ui_grid)
                self.controller.after(0, self.update_tower_viz)
                self.controller.after(0, self.progress_bar.config, {'value': self.controller.blocks_placed})
                self.controller.after(0, self.progress_label.config, {'text': f"{self.controller.blocks_placed} / {len(self.controller.tower_plan)} Blocks"})

                self.controller.after(0, self.update_status, "Robot pressing button to end turn...")

                self.controller.robot.move(PRESS_BUTTON_HOVER_POSE)
                time.sleep(2)
                self.controller.robot.move(PRESS_BUTTON_TOUCH_POSE)
                time.sleep(2)
                self.controller.robot.move(PRESS_BUTTON_HOVER_POSE)
            else:
                self.controller.after(0, self.update_status, "ERROR: Pick sequence failed.")
                break
        
        if not self.controller.stop_threads:
            #FINAL HEIGHT VERIFICATION
            self.controller.after(0, self.update_status, "Tower complete! Verifying final height...")
            self.controller.camera_is_paused.clear()
            time.sleep(0.5)

            is_correct, measured_z, expected_z = verify_tower_height(
                self.controller.robot,
                self.controller.model,
                self.controller.pipeline,
                self.controller.align,
                len(self.controller.tower_plan)
            )

            self.controller.camera_is_paused.set()

            #Printing interface final message
            if is_correct:
                final_message = f"Game Over: Success! Measured height ({measured_z:.3f}m) matches the expected height of {len(self.controller.tower_plan)} blocks."
            else:
                final_message = f"Game Over: Mismatch! Measured height ({measured_z:.3f}m) does not match expected ({expected_z:.3f}m)."
            
            self.controller.after(0, self.update_status, final_message)


def verify_tower_height(robot, model, pipeline, align, expected_block_count):
    print("[INFO] Verifying final tower height...")
    robot.move(INITIAL_SCAN_POSE)
    time.sleep(2)

    stack_pos = find_stack_location(robot, model, pipeline, align)

    if stack_pos is None:
        print("[ERROR] Could not find the stack for height verification.")
        return (False, 0, 0)

    measured_z = stack_pos[2]
    expected_z = TABLE_Z + (expected_block_count * BLOCK_HEIGHT)

    if abs(measured_z - expected_z) <= HEIGHT_TOLERANCE:
        print(f"[SUCCESS] Height matches. Measured: {measured_z:.3f}m, Expected: {expected_z:.3f}m")
        return (True, measured_z, expected_z)
    else:
        print(f"[FAILURE] Height mismatch. Measured: {measured_z:.3f}m, Expected: {expected_z:.3f}m")
        return (False, measured_z, expected_z)

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
                    
                    # Check if the detected position is within the allowed workspace
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

def calculate_coords_with_interpolation(row, col):
    u = col / (GRID_COLS - 1)
    v = row / (GRID_ROWS - 1)
    p_tl, p_tr, p_bl, p_br = GRID_CORNERS["top_left"], GRID_CORNERS["top_right"], GRID_CORNERS["bottom_left"], GRID_CORNERS["bottom_right"]
    top = p_tl + u * (p_tr - p_tl)
    bot = p_bl + u * (p_br - p_bl)
    final = top + v * (bot - top)
    return final[0], final[1]
    
def execute_pick_and_place(robot, pick_x, pick_y, place_z, stack_pos):
    try:
        pick_orientation = {"roll": 2.993, "pitch": 1.430, "yaw": 1.506}
        place_orientation = {"roll": -0.262, "pitch": 1.523, "yaw": 1.228}
        
        hover_pick = PoseObject(x=pick_x, y=pick_y, z=PICK_HOVER_HEIGHT, **pick_orientation)
        robot.move(hover_pick)
        time.sleep(1)
        robot.open_gripper()
        time.sleep(0.5)
        target_pick = PoseObject(x=pick_x, y=pick_y, z=PICK_TARGET_HEIGHT, **pick_orientation)
        robot.move(target_pick)
        time.sleep(1)
        robot.close_gripper()
        time.sleep(1)
        robot.move(hover_pick)
        
        hover_place = PoseObject(x=stack_pos[0], y=stack_pos[1], z=place_z + 0.05, **place_orientation)
        robot.move(hover_place)
        time.sleep(1.5)
        target_place = PoseObject(x=stack_pos[0], y=stack_pos[1], z=place_z, **place_orientation)
        robot.move(target_place)
        time.sleep(1)
        robot.open_gripper()
        time.sleep(1)
        robot.move(hover_place)
        robot.close_gripper()
        return True
    except Exception as e:
        print(f"Pick/place sequence failed: {e}")
        robot.set_learning_mode(True)
        return False

if __name__ == "__main__":
    try:
        model = YOLO("model_4_blocks.pt")
        app = TowerBuilderApp(model)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"A critical error occurred on startup:\n{e}")