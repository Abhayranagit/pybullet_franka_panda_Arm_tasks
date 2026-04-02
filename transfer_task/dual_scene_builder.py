import pybullet as p
import pybullet_data
import numpy as np
import random

def create_dual_arm_scene():
    """
    Loads Plane, 2 Pandas, and a randomly placed Cube.
    Returns: IDs dictionary and the random cube starting position.
    """
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Smaller timestep for more stable physics during grasping
    p.setTimeStep(1./480.)
    
    # 1. Setup Robot 1 (Receiver)
    r1_pos = [0.15, -0.8, 0]
    r1_orn = p.getQuaternionFromEuler([0, 0, 1.57]) 
    r1_id = p.loadURDF("franka_panda/panda.urdf", r1_pos, r1_orn, useFixedBase=True)
    
    # R1 Sticky Fingers
    p.changeDynamics(r1_id, 9, lateralFriction=3.0, spinningFriction=0.5, rollingFriction=0.01)
    p.changeDynamics(r1_id, 10, lateralFriction=3.0, spinningFriction=0.5, rollingFriction=0.01)
    
    # 2. Setup Robot 2 (Sender)
    r2_pos = [0, 0.8, 0]
    r2_orn = p.getQuaternionFromEuler([0, 0, -1.57]) 
    r2_id = p.loadURDF("franka_panda/panda.urdf", r2_pos, r2_orn, useFixedBase=True)
    
    # R2 Sticky Fingers
    p.changeDynamics(r2_id, 9, lateralFriction=3.0, spinningFriction=0.5, rollingFriction=0.01)
    p.changeDynamics(r2_id, 10, lateralFriction=3.0, spinningFriction=0.5, rollingFriction=0.01)

    # 3. Randomize Cube Position (Within R2's reachable workspace)
    rand_x: float = random.uniform(-0.10, 0.20)  
    rand_y = random.uniform(0.45, 0.60)
    cube_start_pos = [rand_x, rand_y, 0.05]
    
    cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, globalScaling=1.0)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    # Lower mass = easier to grip; higher damping = less slipping
    p.changeDynamics(cube_id, -1, 
                        lateralFriction=0.8,      # Realistic friction (not too grippy)
                        spinningFriction=0.2,     # Allows some rotation but stable
                        rollingFriction=0.01,     # Minimal rolling for stability
                        mass=0.5,                 # Heavier mass for realistic weight (500g cube)
                        linearDamping=0.1,        # Slight damping for natural settling
                        angularDamping=0.1,       # Prevents excessive spinning
                        restitution=0.1)          # Low bounce - cubes don't bounce much

    return {'r1': r1_id, 'r2': r2_id, 'cube': cube_id}, np.array(cube_start_pos)