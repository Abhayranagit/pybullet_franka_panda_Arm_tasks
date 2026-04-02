import sys
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import h5py

# Local imports
from single_robot_controller import SinglePandaController
from single_trajectory_planner import SinglePlanner
import single_scene_builder

class DataRecorder:
    def __init__(self):
        self.qpos = []
        self.actions = []
        self.images_high = []
        self.images_wrist = []
        self.images_front = []
        
        # 1. High Camera (Overhead 45 degree angle)
        self.view_matrix_high = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 0.8],     
            cameraTargetPosition=[0.5, 0.0, 0.0],  
            cameraUpVector=[0, 0, 1]
        )
        
        # 2. Front Camera (Looking straight at the boxes from the front)
        self.view_matrix_front = p.computeViewMatrix(
            cameraEyePosition=[0.9, 0.0, 0.15],     # Low and in front of the arm 
            cameraTargetPosition=[0.35, 0.0, 0.05], # Looking directly at the spawn center
            cameraUpVector=[0, 0, 1]
        )

        # Shared Projection Matrix (60 FOV, 640x480)
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=640.0/480.0, nearVal=0.01, farVal=10.0
        )

    def record_step(self, robot_id, target_joints, target_gripper):
        # --- 3. Dynamic Wrist Camera ---
        # Get the real-time position of the end-effector (link index 11)
        ee_state = p.getLinkState(robot_id, 11)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1] # Quaternion
        
        # Calculate the forward vector (where the gripper is pointing)
        rot_matrix = p.getMatrixFromQuaternion(ee_orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Assuming the gripper points along its local Z-axis (common for Franka)
        forward_vec = rot_matrix[:, 2] 
        up_vec = rot_matrix[:, 0] # Local X-axis usually serves as 'up' for the camera
        
        # Place camera slightly behind the fingertips so it doesn't clip through the object
        cam_pos = ee_pos - (forward_vec * 0.05) 
        cam_target = ee_pos + (forward_vec * 0.1) # Look 10cm ahead of the fingers
        
        view_matrix_wrist = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=up_vec
        )

        # --- Capture Images ---
        # High Cam
        _, _, rgb_high, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=self.view_matrix_high,
            projectionMatrix=self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        self.images_high.append(np.reshape(rgb_high, (480, 640, 4))[:, :, :3])

        # Front Cam
        _, _, rgb_front, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=self.view_matrix_front,
            projectionMatrix=self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        self.images_front.append(np.reshape(rgb_front, (480, 640, 4))[:, :, :3])

        # Wrist Cam
        _, _, rgb_wrist, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=view_matrix_wrist,
            projectionMatrix=self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        self.images_wrist.append(np.reshape(rgb_wrist, (480, 640, 4))[:, :, :3])

        # --- Record Kinematics ---
        joint_states = p.getJointStates(robot_id, [0, 1, 2, 3, 4, 5, 6])
        current_qpos = [state[0] for state in joint_states]
        self.qpos.append(current_qpos)

        action = list(target_joints) + [target_gripper]
        self.actions.append(action)

    def save_to_disk(self, filename):
        print(f"Saving {len(self.qpos)} synchronized steps to {filename}...")
        with h5py.File(filename, 'w') as f:
            obs = f.create_group('observations')
            obs.create_dataset('qpos', data=np.array(self.qpos, dtype=np.float32))
            
            img_grp = obs.create_group('images')
            # Save all three camera streams using fast compression
            img_grp.create_dataset('cam_high', data=np.array(self.images_high, dtype=np.uint8), compression="gzip")
            img_grp.create_dataset('cam_front', data=np.array(self.images_front, dtype=np.uint8), compression="gzip")
            img_grp.create_dataset('cam_wrist', data=np.array(self.images_wrist, dtype=np.uint8), compression="gzip")
            
            f.create_dataset('action', data=np.array(self.actions, dtype=np.float32))
            f.attrs['sim'] = True
            f.attrs['num_samples'] = len(self.qpos)
        print("Dataset saved successfully!\n")

def run_episode(episode_idx):
    print(f"=====================================")
    print(f"--- STARTING EPISODE {episode_idx} ---")
    print(f"=====================================")
    
    # Reset simulation state completely for the new episode
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0.5, 0, 0])

    robot_id, cube_ids = single_scene_builder.create_single_arm_scene()
    controller = SinglePandaController(robot_id, cube_ids)
    
    for _ in range(100): p.stepSimulation()
        
    cubes_data = []
    target_cubes = ["cube_1", "cube_3", "cube_4"]
    for name in target_cubes:
        if name in cube_ids:
            box_id = cube_ids[name]
            pos, orn = p.getBasePositionAndOrientation(box_id)
            euler = p.getEulerFromQuaternion(orn)
            cubes_data.append({'name': name, 'pos': list(pos), 'yaw': euler[2]})

    planner = SinglePlanner(controller)
    target_xy = np.array([0.6, 0.0])

    trajectory = planner.generate_full_stack_mission(target_xy, cubes_data)
    
    recorder = DataRecorder()
    trajectory_idx = 0
    physics_step_counter = 0
    RECORD_INTERVAL = 10 
    gripper_state = 0.04 
    last_action = ""

    while p.isConnected():
        if trajectory_idx < len(trajectory):
            step = trajectory[trajectory_idx]
            action = step["action"]
            
            if any(k in action for k in ["grasp", "lift", "transfer", "stack"]):
                gripper_state = -0.01 
            else: 
                gripper_state = 0.04  

            # --- THE STRICT RELEASE CONDITION ---
            # If the planner is trying to 'retreat' to get the next box, 
            # but the fingers haven't physically opened yet, HALT THE ARM!
            if "retreat" in action and not controller.is_gripper_fully_open():
                # Command the gripper to open, but DO NOT move the arm or advance the trajectory
                controller.set_commands(step["joints"], 0.04)
                p.stepSimulation()
                continue # This skips traj_idx += 1, effectively pausing time!

            controller.set_commands(step["joints"], gripper_state)
            
            if physics_step_counter % RECORD_INTERVAL == 0:
                recorder.record_step(robot_id, step["joints"], gripper_state)
            
            if action != last_action:
                print(f"Executing: {action}")
                last_action = action
            
            trajectory_idx += 1
        else:
            if last_action != "DONE":
                print("--- STACKING COMPLETE ---")
                recorder.save_to_disk(f"episode_{episode_idx}.hdf5")
                break 

        p.stepSimulation()
        physics_step_counter += 1

if __name__ == "__main__":
    # --- PRO TIP: HEADLESS MODE ---
    # Change p.GUI to p.DIRECT below to generate the data 10x faster!
    # Without the GUI window, PyBullet calculates the physics instantly.
    physicsClient = p.connect(p.GUI) 
    
    # Loop exactly 50 times
    TOTAL_EPISODES = 50
    for i in range(TOTAL_EPISODES):
        run_episode(i)
        
    p.disconnect()
    print("--- ALL EPISODES COMPLETED SUCCESSFULLY ---")