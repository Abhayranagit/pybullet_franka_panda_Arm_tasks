import os
import glob
import h5py
from PIL import Image
import argparse

def extract_final_front_views(data_dir, output_dir):
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .hdf5 files in the data directory
    search_pattern = os.path.join(data_dir, "*.hdf5")
    hdf5_files = glob.glob(search_pattern)
    
    if not hdf5_files:
        print(f"Error: No .hdf5 files found in '{data_dir}'.")
        return

    print(f"Found {len(hdf5_files)} episodes. Extracting final front views...")

    extracted_count = 0
    
    # Sort files naturally (optional, but keeps them in order)
    for file_path in sorted(hdf5_files):
        filename = os.path.basename(file_path)
        episode_name = os.path.splitext(filename)[0] # e.g., 'episode_0'

        try:
            with h5py.File(file_path, 'r') as f:
                # Ensure the new camera actually exists in this file
                if 'cam_front' in f['observations']['images']:
                    
                    # Grab the VERY LAST frame of the episode (index -1)
                    final_image_array = f['observations']['images']['cam_front'][-1]
                    
                    # Convert the numpy array to a standard PNG image and save
                    img = Image.fromarray(final_image_array)
                    save_path = os.path.join(output_dir, f"{episode_name}_result.png")
                    img.save(save_path)
                    
                    extracted_count += 1
                else:
                    print(f"Warning: 'cam_front' not found in {filename}. (Is this an old recording?)")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"\n{'='*50}")
    print(f"✅ Success! Extracted {extracted_count} final frames.")
    print(f"📁 Open the '{output_dir}' folder to review your stacking results.")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the final front-camera frame from all recorded episodes to verify stacking success.")
    parser.add_argument("--data_dir", type=str, default="data", help="Folder containing your episode_X.hdf5 files.")
    parser.add_argument("--output_dir", type=str, default="dataset_review", help="Folder to save the extracted images.")
    
    args = parser.parse_args()
    extract_final_front_views(args.data_dir, args.output_dir)