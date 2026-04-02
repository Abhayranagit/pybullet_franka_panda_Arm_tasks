import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import h5py

# Local Imports
import dual_scene_builder
from dual_robot_controller import DualPandaController
from dual_trajectory_planner import DualPlanner

class DualDataRecorder:
    def __init__(self):
        self.qpos = []
        self.actions = []
        self.images = []
        
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.5, 0.0, 1.2],     
            cameraTargetPosition=[0.15, 0.0, 0.5], 
            cameraUpVector=[0, 0, 1]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=640.0/480.0, nearVal=0.1, farVal=10.0
        )

    def record_step(self, r1_id, r2_id, r1_tgt_joints, r1_tgt_grip, r2_tgt_joints, r2_tgt_grip):
        w, h, rgb, depth, seg = p.getCameraImage(
            width=640, height=480,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL 
        )
        rgb_array = np.reshape(rgb, (480, 640, 4))[:, :, :3]
        self.images.append(rgb_array)

        # Joint States (14D)
        r1_states = p.getJointStates(r1_id, [0, 1, 2, 3, 4, 5, 6])
        r2_states = p.getJointStates(r2_id, [0, 1, 2, 3, 4, 5, 6])
        r1_qpos = [state[0] for state in r1_states]
        r2_qpos = [state[0] for state in r2_states]
        self.qpos.append(r1_qpos + r2_qpos)

        # Combined Actions (16D)
        action = list(r1_tgt_joints) + [r1_tgt_grip] + list(r2_tgt_joints) + [r2_tgt_grip]
        self.actions.append(action)

    def save_to_disk(self, filename):
        print(f"Saving {len(self.qpos)} synchronized steps to {filename}...")
        with h5py.File(filename, 'w') as f:
            obs = f.create_group('observations')
            obs.create_dataset('qpos', data=np.array(self.qpos, dtype=np.float32))
            
            img_grp = obs.create_group('images')
            img_grp.create_dataset('cam_high', data=np.array(self.images, dtype=np.uint8), compression="gzip")
            
            f.create_dataset('action', data=np.array(self.actions, dtype=np.float32))
            f.attrs['sim'] = True
            f.attrs['num_samples'] = len(self.qpos)

def run_episode(episode_idx):
    print(f"\n=====================================")
    print(f"--- STARTING EPISODE {episode_idx} ---")
    print(f"=====================================")
    
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.5])

    # Build Scene (Returns dict and randomized cube pos)
    ids_dict, cube_pos = dual_scene_builder.create_dual_arm_scene()
    controller = DualPandaController(ids_dict)
    
    for _ in range(50): p.stepSimulation()

    planner = DualPlanner(controller)
    trajectory = planner.generate_handover_mission(cube_pos)
    
    recorder = DualDataRecorder()
    traj_idx = 0
    physics_step_counter = 0
    RECORD_INTERVAL = 10 
    
    r1_grip = 0.04 
    r2_grip = 0.04 
    last_action = ""

    while p.isConnected():
        if traj_idx < len(trajectory):
            step = trajectory[traj_idx]
            action = step["action"]
            
            # --- STATE MACHINE ---
            if action == "r2_close":
                r2_grip = -0.01 
            elif action in ["r2_move_open", "r2_descend", "r2_plunge"]:
                r2_grip = 0.04  
            elif action in ["r2_lift", "staging_move", "r1_approach"]:
                r2_grip = -0.01
                r1_grip = 0.04 
            elif action == "r1_grasp":
                r1_grip = -0.01 
                r2_grip = -0.01 
            elif action == "r2_release":
                r2_grip = 0.04  
                r1_grip = -0.01 
            elif action in ["r2_retreat", "r1_retreat"]:
                r2_grip = 0.04
                r1_grip = -0.01

            controller.set_commands(step["r1"], r1_grip, step["r2"], r2_grip)
            
            if physics_step_counter % RECORD_INTERVAL == 0:
                recorder.record_step(
                    ids_dict['r1'], ids_dict['r2'], 
                    step["r1"], r1_grip, 
                    step["r2"], r2_grip
                )
            
            traj_idx += 1
        else:
            print("--- MISSION COMPLETE ---")
            filepath = os.path.join("episode", f"dual_episode_{episode_idx}.hdf5")
            recorder.save_to_disk(filepath)
            break
        
        p.stepSimulation()
        physics_step_counter += 1

if __name__ == "__main__":
    os.makedirs("episode", exist_ok=True)
    
    # Change p.GUI to p.DIRECT for incredibly fast, headless data collection
    p.connect(p.GUI)
    
    TOTAL_EPISODES = 50
    for i in range(TOTAL_EPISODES):
        run_episode(i)
        
    p.disconnect()
    print("\n--- ALL DATASET RECORDING COMPLETE ---")