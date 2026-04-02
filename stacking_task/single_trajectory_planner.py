import numpy as np
import pybullet as p

class SinglePlanner:
    def __init__(self, controller):
        self.controller = controller
        self.robot_id = controller.robot_id
        self.ee_idx = controller.ee_link_index

    def solve_ik(self, target_pos, target_yaw=0.0, current_joints=None):
        target_orn = p.getQuaternionFromEuler([np.pi, 0, target_yaw]) 
        
        # FIX 1: USE SEED JOINTS
        # By telling the math solver where the arm currently is, it prevents 
        # the elbow from "twitching" or jumping between frames.
        rest = current_joints if current_joints else [0, -0.78, 0, -2.35, 0, 1.57, 0.78]
        
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, 
            self.ee_idx, 
            target_pos, 
            target_orn,
            lowerLimits=[-2.9]*7, 
            upperLimits=[2.9]*7, 
            jointRanges=[5.8]*7, 
            restPoses=rest,
            maxNumIterations=200,
            residualThreshold=1e-5
        )
        return list(joint_poses[:7]) 

    def interpolate_segment(self, start_pos, end_pos, steps, action_name, plan_list, start_yaw=0.0, end_yaw=0.0):
        # Fetch the last known joints to use as our seed
        prev_q = plan_list[-1]["joints"] if plan_list else None
        
        for t in np.linspace(0, 1, int(steps)):
            # FIX 2: COSINE EASING
            # This creates a perfect Bell Curve for speed. The arm starts slowly,
            # speeds up in the middle, and glides smoothly to a stop. No whiplash!
            smooth_t = (1 - np.cos(t * np.pi)) / 2 
            
            curr_pos = start_pos * (1-smooth_t) + end_pos * smooth_t
            curr_yaw = start_yaw * (1-smooth_t) + end_yaw * smooth_t
            
            q = self.solve_ik(curr_pos, curr_yaw, current_joints=prev_q)
            prev_q = q # Save for the next micro-step
            
            plan_list.append({"joints": q, "action": action_name})

    def generate_full_stack_mission(self, target_xy, cubes_data):
        full_plan = []
        HOVER_HEIGHT = 0.35
        
        start_state = p.getLinkState(self.robot_id, self.ee_idx)
        HOME_POS = np.array(start_state[0])
        HOME_YAW = 0.0 
        
        CUBE_HEIGHT = 0.07 
        TCP_OFFSET = -0.01  
        stack_top_z = 0.0   
        
        current_pos = HOME_POS
        current_yaw = HOME_YAW
        
        self.interpolate_segment(HOME_POS, HOME_POS, 10, "home", full_plan, start_yaw=HOME_YAW, end_yaw=HOME_YAW)

        for i, cube_info in enumerate(cubes_data):
            cube_num = i + 1
            print(f"--- PLANNING CUBE {cube_num} ---")
            
            src_pos = np.array(cube_info['pos'])
            src_yaw = cube_info['yaw'] 
            
            src_hover = np.array([src_pos[0], src_pos[1], HOVER_HEIGHT])
            src_grasp = np.array([src_pos[0], src_pos[1], src_pos[2] + TCP_OFFSET])
            tgt_hover = np.array([target_xy[0], target_xy[1], HOVER_HEIGHT])
            
            target_center_z = stack_top_z + (CUBE_HEIGHT / 2.0)
            place_z = target_center_z + TCP_OFFSET + 0.002 
            tgt_place = np.array([target_xy[0], target_xy[1], place_z]) 

            # --- FIX 3: SLOW-MOTION STEP COUNTS ---
            # Increased all movement durations by 3x to mimic real-life speed
            self.interpolate_segment(current_pos, src_hover, 200, f"move_to_c{cube_num}", full_plan, start_yaw=current_yaw, end_yaw=src_yaw)
            self.interpolate_segment(src_hover, src_grasp, 150, f"descend_c{cube_num}", full_plan, start_yaw=src_yaw, end_yaw=src_yaw)
            
            # Huge wait time to ensure physical squeeze is solid before lifting
            last_q = full_plan[-1]['joints']
            for _ in range(150): full_plan.append({"joints": last_q, "action": "grasp"})
            
            self.interpolate_segment(src_grasp, src_hover, 150, f"lift_c{cube_num}", full_plan, start_yaw=src_yaw, end_yaw=src_yaw)

            # --- PLACE SEQUENCE ---
            stack_yaw = 0.0 
            self.interpolate_segment(src_hover, tgt_hover, 250, f"transfer_c{cube_num}", full_plan, start_yaw=src_yaw, end_yaw=stack_yaw)
            
            last_q = full_plan[-1]['joints']
            for _ in range(50): full_plan.append({"joints": last_q, "action": f"transfer_c{cube_num}"})
            
            self.interpolate_segment(tgt_hover, tgt_place, 150, f"stack_c{cube_num}", full_plan, start_yaw=stack_yaw, end_yaw=stack_yaw)
            
            # Huge wait time to ensure fingers peel completely off the box
            last_q = full_plan[-1]['joints']
            for _ in range(200): full_plan.append({"joints": last_q, "action": "release"})
            
            self.interpolate_segment(tgt_place, tgt_hover, 150, "retreat", full_plan, start_yaw=stack_yaw, end_yaw=stack_yaw)
            
            current_pos = tgt_hover
            current_yaw = stack_yaw
            stack_top_z += CUBE_HEIGHT 

        self.interpolate_segment(current_pos, HOME_POS, 200, "return_home", full_plan, start_yaw=current_yaw, end_yaw=HOME_YAW)

        return full_plan