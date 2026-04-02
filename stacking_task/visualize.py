import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_trajectory_at_step(hdf5_path, timestep):
    if not os.path.exists(hdf5_path):
        print(f"Error: File '{hdf5_path}' not found.")
        return

    with h5py.File(hdf5_path, 'r') as f:
        num_samples = f.attrs['num_samples']
        
        # Handle negative indexing (e.g., -1 for the last step)
        if timestep < 0:
            timestep = num_samples + timestep

        if timestep >= num_samples or timestep < 0:
            print(f"Error: Timestep {timestep} is out of bounds (0 to {num_samples - 1}).")
            return

        # 1. Extract FULL trajectories for the graphs
        qpos_full = f['observations']['qpos'][:]       # Shape: (T, 7)
        action_full = f['action'][:]                   # Shape: (T, 8)
        
        # 2. Extract IMAGES for the specific timestep
        img_high = f['observations']['images']['cam_high'][timestep]
        img_front = f['observations']['images']['cam_front'][timestep]
        img_wrist = f['observations']['images']['cam_wrist'][timestep]

        # --- Plotting Setup ---
        # Create a large figure with a grid layout
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Trajectory & Vision Analysis - File: {os.path.basename(hdf5_path)} | Timestep: {timestep}", fontsize=16)

        # --- TOP ROW: Camera Images ---
        ax_img1 = plt.subplot(2, 3, 1)
        ax_img1.imshow(img_high)
        ax_img1.set_title('cam_high (Overhead)')
        ax_img1.axis('off')

        ax_img2 = plt.subplot(2, 3, 2)
        ax_img2.imshow(img_front)
        ax_img2.set_title('cam_front (Front)')
        ax_img2.axis('off')

        ax_img3 = plt.subplot(2, 3, 3)
        ax_img3.imshow(img_wrist)
        ax_img3.set_title('cam_wrist (Egocentric)')
        ax_img3.axis('off')

        # --- BOTTOM ROW: Trajectory Graphs ---
        time_axis = np.arange(num_samples)

        # Graph 1: Arm Joints (qpos vs action)
        ax_arm = plt.subplot(2, 2, 3)
        ax_arm.set_title('Arm Joints Trajectory (0-6)')
        # Plot only a couple of joints to keep it readable (e.g., Joint 0 and Joint 4)
        ax_arm.plot(time_axis, qpos_full[:, 0], label='qpos J0 (Base)', color='blue')
        ax_arm.plot(time_axis, action_full[:, 0], label='action J0', color='blue', linestyle=':')
        ax_arm.plot(time_axis, qpos_full[:, 4], label='qpos J4 (Wrist)', color='green')
        ax_arm.plot(time_axis, action_full[:, 4], label='action J4', color='green', linestyle=':')
        
        ax_arm.axvline(x=timestep, color='red', linestyle='--', linewidth=2, label='Current Timestep')
        ax_arm.set_xlabel('Timestep')
        ax_arm.set_ylabel('Radian Angle')
        ax_arm.legend(loc='upper right')
        ax_arm.grid(True, alpha=0.3)

        # Graph 2: Gripper Action
        ax_grip = plt.subplot(2, 2, 4)
        ax_grip.set_title('Gripper Action Command')
        # Plot the gripper action (index 7)
        ax_grip.plot(time_axis, action_full[:, 7], label='Gripper Target', color='purple', linewidth=2)
        
        ax_grip.axvline(x=timestep, color='red', linestyle='--', linewidth=2, label='Current Timestep')
        ax_grip.set_xlabel('Timestep')
        ax_grip.set_ylabel('Gripper State (Open/Close)')
        ax_grip.legend(loc='upper right')
        ax_grip.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit title
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot trajectory graphs and show images for a specific timestep.")
    parser.add_argument("file", type=str, help="Path to the .hdf5 episode file (e.g., data/episode_0.hdf5)")
    parser.add_argument("step", type=int, help="The timestep index to visualize.")
    
    args = parser.parse_args()
    visualize_trajectory_at_step(args.file, args.step)