import pybullet as p
import pybullet_data
import os
import random
import numpy as np

def create_single_arm_scene():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0])

    robot_start_pos = [0, 0, 0]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("franka_panda/panda.urdf", robot_start_pos, robot_start_orn, useFixedBase=True)

    # --- THE STICKY FINGER FIX ---
    # Lowered from 3.0 to 1.5. Grippy, but not sticky!
    p.changeDynamics(robot_id, 9, lateralFriction=1.5, spinningFriction=0.1)
    p.changeDynamics(robot_id, 10, lateralFriction=1.5, spinningFriction=0.1)
    
    cubes_config = [
        {'name': 'cube_1', 'rgba': [1, 0, 0, 1]},
        {'name': 'cube_3', 'rgba': [0, 0, 1, 1]},
        {'name': 'cube_4', 'rgba': [1, 1, 0, 1]},
    ]

    x_range = [0.25, 0.5]  
    y_range = [-0.35, 0.35] 

    cube_ids = {}
    placed_positions = []

    for c in cubes_config:
        valid_position = False
        candidate_pos = None
        
        while not valid_position:
            rx = random.uniform(x_range[0], x_range[1])
            ry = random.uniform(y_range[0], y_range[1])
            # Position cubes properly on the ground - cube_small.urdf scaled by 1.4 = ~0.07m high
            # So half-height = 0.035m to place bottom edge on ground plane
            candidate_pos = np.array([rx, ry, 0.035])
            
            if not placed_positions:
                valid_position = True
            else:
                distances = [np.linalg.norm(candidate_pos[:2] - p_pos[:2]) for p_pos in placed_positions]
                if min(distances) > 0.08: 
                    valid_position = True
                    
        placed_positions.append(candidate_pos)
        
        c_id = p.loadURDF("cube_small.urdf", candidate_pos.tolist(), globalScaling=1.4)
        p.changeVisualShape(c_id, -1, rgbaColor=c['rgba'])
        
        # Real-world physics parameters for heavy, stable cubes
        p.changeDynamics(c_id, -1, 
                        lateralFriction=0.8,      # Realistic friction (not too grippy)
                        spinningFriction=0.2,     # Allows some rotation but stable
                        rollingFriction=0.01,     # Minimal rolling for stability
                        mass=0.5,                 # Heavier mass for realistic weight (500g cube)
                        linearDamping=0.1,        # Slight damping for natural settling
                        angularDamping=0.1,       # Prevents excessive spinning
                        restitution=0.1)          # Low bounce - cubes don't bounce much
        cube_ids[c['name']] = c_id

    # --- VISUAL BOUNDING BOX 1: SPAWN ZONE (NEON GREEN) ---
    bbox_z = 0.001 # Slightly above the ground plane to prevent z-fighting
    p1 = [x_range[0], y_range[0], bbox_z] # Bottom Left
    p2 = [x_range[1], y_range[0], bbox_z] # Top Left
    p3 = [x_range[1], y_range[1], bbox_z] # Top Right
    p4 = [x_range[0], y_range[1], bbox_z] # Bottom Right
    
    neon_green = [0, 1, 0]
    line_width = 3
    
    p.addUserDebugLine(p1, p2, lineColorRGB=neon_green, lineWidth=line_width)
    p.addUserDebugLine(p2, p3, lineColorRGB=neon_green, lineWidth=line_width)
    p.addUserDebugLine(p3, p4, lineColorRGB=neon_green, lineWidth=line_width)
    p.addUserDebugLine(p4, p1, lineColorRGB=neon_green, lineWidth=line_width)

    # --- VISUAL BOUNDING BOX 2: TARGET STACKING ZONE (NEON CYAN) ---
    target_x = 0.6
    target_y = 0.0
    box_half_size = 0.1 # Creates a 20cm x 20cm box around the target
    
    t1 = [target_x - box_half_size, target_y - box_half_size, bbox_z]
    t2 = [target_x + box_half_size, target_y - box_half_size, bbox_z]
    t3 = [target_x + box_half_size, target_y + box_half_size, bbox_z]
    t4 = [target_x - box_half_size, target_y + box_half_size, bbox_z]
    
    neon_cyan = [0, 1, 1]
    
    p.addUserDebugLine(t1, t2, lineColorRGB=neon_cyan, lineWidth=line_width)
    p.addUserDebugLine(t2, t3, lineColorRGB=neon_cyan, lineWidth=line_width)
    p.addUserDebugLine(t3, t4, lineColorRGB=neon_cyan, lineWidth=line_width)
    p.addUserDebugLine(t4, t1, lineColorRGB=neon_cyan, lineWidth=line_width)

    return robot_id, cube_ids