print("hello")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

def extract_timestep(hdf5_path, timestep, output_dir=None):
    if not os.path.exists(hdf5_path):
        print(f"Error: File '{hdf5_path}' not found.")
        return

    with h5py.File(hdf5_path, 'r') as f:
        num_samples = f.attrs['num_samples']
        
        # Handle negative indexing (e.g., -1 for the last step)
        if timestep < 0:
            timestep = num_samples + timestep

        # Boundary check
        if timestep >= num_samples or timestep < 0:
            print(f"Error: Timestep {timestep} is out of bounds.")
            print(f"This episode contains {num_samples} steps (valid indices: 0 to {num_samples - 1}).")
            return

        # 1. Extract Kinematics & Actions
        qpos = f['observations']['qpos'][timestep]
        action = f['action'][timestep]

        print(f"\n{'='*50}")
        print(f" DATA FOR TIMESTEP: {timestep} (File: {os.path.basename(hdf5_path)})")
        print(f"{'='*50}")
        print(f"Proprioception (qpos):")
        print(f"  Arm Joints : {np.round(qpos[:7], 4)}")
        print(f"\nCommanded Action (target):")
        print(f"  Arm Joints : {np.round(action[:7], 4)}")
        print(f"  Gripper    : {np.round(action[7], 4)}")
        print(f"{'='*50}\n")

        # 2. Extract Images
        cam_high = f['observations']['images']['cam_high'][timestep]
        cam_front = f['observations']['images']['cam_front'][timestep]
        cam_wrist = f['observations']['images']['cam_wrist'][timestep]

        # 3. Visualize and/or Save
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cam_high)
        axes[0].set_title('cam_high (Exocentric Overhead)')
        axes[0].axis('off')

        axes[1].imshow(cam_front)
        axes[1].set_title('cam_front (Exocentric Front)')
        axes[1].axis('off')

        axes[2].imshow(cam_wrist)
        axes[2].set_title('cam_wrist (Egocentric Wrist)')
        axes[2].axis('off')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the combined plot
            plot_path = os.path.join(output_dir, f'episode_{os.path.basename(hdf5_path)}_step_{timestep}_combined.png')
            plt.savefig(plot_path, dpi=150)
            
            # Save the individual raw image files
            Image.fromarray(cam_high).save(os.path.join(output_dir, f'step_{timestep}_cam_high.png'))
            Image.fromarray(cam_front).save(os.path.join(output_dir, f'step_{timestep}_cam_front.png'))
            Image.fromarray(cam_wrist).save(os.path.join(output_dir, f'step_{timestep}_cam_wrist.png'))
            
            print(f"Images successfully saved to: {output_dir}/")
        else:
            print("Opening image viewer... (Close the window to exit script)")
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a specific timestep from an ACT HDF5 demonstration.")
    parser.add_argument("file", type=str, help="Path to the .hdf5 episode file (e.g., data/episode_0.hdf5)")
    parser.add_argument("step", type=int, help="The timestep index to extract (e.g., 50. Use -1 for the final step).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the extracted images (optional).")
    
    args = parser.parse_args()
    
    extract_timestep(args.file, args.step, args.save_dir)