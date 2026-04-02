import numpy as np
import pybullet as p

class DualPlanner:
    def __init__(self, controller):
        self.r1_id = controller.r1_id
        self.r2_id = controller.r2_id
        self.ee_idx = controller.ee_idx
        
        self.orn_down = p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        self.orn_r2_horiz = p.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])
        self.orn_r1_vert = p.getQuaternionFromEuler([np.pi/2, 0, np.pi])

    def solve_ik(self, robot_id, target_pos, orient_mode, seed_joints=None):
        if orient_mode == "r2_present_horizontal": target_orn = self.orn_r2_horiz
        elif orient_mode == "r1_receive_vertical": target_orn = self.orn_r1_vert
        else: target_orn = self.orn_down

        rest = seed_joints if seed_joints else [0, -0.78, 0, -2.35, 0, 1.57, 0.78]

        joint_poses = p.calculateInverseKinematics(
            robot_id, self.ee_idx, target_pos, target_orn,
            lowerLimits=[-2.9]*7, upperLimits=[2.9]*7, 
            jointRanges=[5.8]*7, restPoses=rest,
            maxNumIterations=200,
            residualThreshold=1e-5
        )
        return list(joint_poses[:7])

    def interpolate_segment(self, steps, r1_args, r2_args, action_name, plan_list):
        """Strict Cartesian linear interpolation. Clamping removed for perfect accuracy."""
        steps = int(steps)
        prev_q1, prev_q2 = None, None
        
        r1_o = r1_args.get('orient', 'down')
        r2_o = r2_args.get('orient', 'down')
        
        for t in np.linspace(0, 1, steps):
            curr_r1_pos = r1_args['start'] * (1-t) + r1_args['end'] * t
            curr_r2_pos = r2_args['start'] * (1-t) + r2_args['end'] * t
            
            # Utilizing seed_joints ensures smooth transitions without needing artificial clamps
            q1 = self.solve_ik(self.r1_id, curr_r1_pos, r1_o, seed_joints=prev_q1)
            q2 = self.solve_ik(self.r2_id, curr_r2_pos, r2_o, seed_joints=prev_q2)
            
            prev_q1, prev_q2 = q1, q2
            plan_list.append({"r1": q1, "r2": q2, "action": action_name})

    def interpolate_joint_segment(self, steps, r1_args, r2_args, action_name, plan_list):
        r1_start_o = r1_args.get('start_orient', r1_args.get('orient'))
        r1_end_o   = r1_args.get('end_orient',   r1_args.get('orient'))
        r2_start_o = r2_args.get('start_orient', r2_args.get('orient'))
        r2_end_o   = r2_args.get('end_orient',   r2_args.get('orient'))

        q1_start = np.array(self.solve_ik(self.r1_id, r1_args['start'], r1_start_o))
        q1_end   = np.array(self.solve_ik(self.r1_id, r1_args['end'],   r1_end_o, seed_joints=q1_start.tolist()))
        q2_start = np.array(self.solve_ik(self.r2_id, r2_args['start'], r2_start_o))
        q2_end   = np.array(self.solve_ik(self.r2_id, r2_args['end'],   r2_end_o, seed_joints=q2_start.tolist()))

        for t in np.linspace(0, 1, int(steps)):
            smooth_t = (1 - np.cos(t * np.pi)) / 2  
            q1_curr = q1_start * (1-smooth_t) + q1_end * smooth_t
            q2_curr = q2_start * (1-smooth_t) + q2_end * smooth_t
            plan_list.append({"r1": q1_curr.tolist(), "r2": q2_curr.tolist(), "action": action_name})

    def generate_handover_mission(self, cube_pos):
        full_plan = []
        CUBE_HEIGHT = 0.05
        r2_home_pos = np.array([0.0, 0.3, 0.5]) 
        r1_home_pos = np.array([0.15, -0.3, 0.5])
        TCP_OFFSET = -0.020

        # --- FIXED: Proper grasp positioning ---
        PRE_GRASP_OFFSET = 0.03  # Safely hover 6cm above the cube
        
        # For grasping, we want to grip at the cube center or slightly below
        # cube_pos[2] is already the cube center height
        GRASP_OFFSET = TCP_OFFSET  # Just use the TCP offset, don't add extra height
    
        r2_hover = cube_pos + np.array([0, 0, 0.25])
        r2_pre_grasp = cube_pos + np.array([0, 0, PRE_GRASP_OFFSET])
        r2_grasp = cube_pos + np.array([0, 0, GRASP_OFFSET])  # This will grasp at cube center
    
        TRANSFER_HEIGHT = 0.55
        R1_X = 0.15
        transfer_center = np.array([R1_X, 0.0, TRANSFER_HEIGHT]) 
        
        r2_transfer_target = transfer_center
        r1_transfer_target = transfer_center 

        # ====== PHASE 1: R2 PICK ======
        self.interpolate_joint_segment(100, 
            {'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            {'start': r2_home_pos, 'end': r2_hover, 'orient': 'down'}, "r2_move_open", full_plan)
            
        self.interpolate_segment(80,
            {'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            {'start': r2_hover, 'end': r2_pre_grasp, 'orient': 'down'}, "r2_descend", full_plan)
            
        self.interpolate_segment(100,
            {'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            {'start': r2_pre_grasp, 'end': r2_grasp, 'orient': 'down'}, "r2_plunge", full_plan)
        
        last = full_plan[-1]
        for _ in range(80): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r2_plunge"})
            
        for _ in range(100): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r2_close"})
        
        self.interpolate_segment(80,
            {'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            {'start': r2_grasp, 'end': r2_hover, 'orient': 'down'}, "r2_lift", full_plan)

        # ====== PHASE 2: STAGING ======
        self.interpolate_joint_segment(150,
            r1_args={'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            r2_args={'start': r2_hover, 'end': r2_home_pos, 'orient': 'down'},
            action_name="staging_move", plan_list=full_plan)
            
        self.interpolate_joint_segment(200,
            r1_args={'start': r1_home_pos, 'end': r1_home_pos, 'orient': 'down'},
            r2_args={'start': r2_home_pos, 'end': r2_transfer_target, 'orient': 'r2_present_horizontal'},
            action_name="staging_move", plan_list=full_plan)
        
        last = full_plan[-1]
        for _ in range(100): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "staging_move"})

        # ====== PHASE 3: R1 PERFECT RAIL APPROACH ======
        r1_pre_approach = np.array([R1_X, -0.20, TRANSFER_HEIGHT])
        
        self.interpolate_joint_segment(180,
            r1_args={'start': r1_home_pos, 'end': r1_pre_approach, 'start_orient': 'down', 'end_orient': 'r1_receive_vertical'},
            r2_args={'start': r2_transfer_target, 'end': r2_transfer_target, 'orient': 'r2_present_horizontal'},
            action_name="r1_approach", plan_list=full_plan)

        r1_approach_waypoints = [
            np.array([R1_X, -0.20, TRANSFER_HEIGHT]),
            np.array([R1_X, -0.17, TRANSFER_HEIGHT]),
            np.array([R1_X, -0.15, TRANSFER_HEIGHT]),
            np.array([R1_X, -0.13, TRANSFER_HEIGHT]),
            np.array([R1_X, -0.12, TRANSFER_HEIGHT]),
            r1_transfer_target 
        ]
        
        current_r1_pos = r1_pre_approach
        for next_pos in r1_approach_waypoints[1:]:
            self.interpolate_segment(80, 
                r1_args={'start': current_r1_pos, 'end': next_pos, 'orient': 'r1_receive_vertical'},
                r2_args={'start': r2_transfer_target, 'end': r2_transfer_target, 'orient': 'r2_present_horizontal'},
                action_name="r1_approach", plan_list=full_plan)
            current_r1_pos = next_pos
            
        last = full_plan[-1]
        for _ in range(80): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r1_approach"})
        
        # ====== PHASE 4: COORDINATED HANDOVER ======
        last = full_plan[-1]
        for _ in range(120): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r1_grasp"})
        for _ in range(100): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r1_grasp"})
        for _ in range(120): full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r2_release"})

        # ====== PHASE 5: R2 SLIDES OUT FIRST ======
        r2_slide_out_pt = r2_transfer_target + np.array([0.0, 0.15, 0.0]) 
        
        self.interpolate_segment(120,
            r1_args={'start': r1_transfer_target, 'end': r1_transfer_target, 'orient': 'r1_receive_vertical'},
            r2_args={'start': r2_transfer_target, 'end': r2_slide_out_pt, 'orient': 'r2_present_horizontal'},
            action_name="r2_retreat", plan_list=full_plan)

        r2_retreat_pt = r2_slide_out_pt + np.array([0, 0.15, 0.2])
        self.interpolate_joint_segment(180,
            r1_args={'start': r1_transfer_target, 'end': r1_transfer_target, 'orient': 'r1_receive_vertical'},
            r2_args={'start': r2_slide_out_pt, 'end': r2_retreat_pt, 'orient': 'r2_present_horizontal'},
            action_name="r2_retreat", plan_list=full_plan)

        # ====== PHASE 6: R1 RETREAT ======
        r1_retreat_waypoints = r1_approach_waypoints[::-1]
        current_r1_pos = r1_transfer_target
        
        for next_pos in r1_retreat_waypoints[1:]:
            self.interpolate_segment(40,
                r1_args={'start': current_r1_pos, 'end': next_pos, 'orient': 'r1_receive_vertical'},
                r2_args={'start': r2_retreat_pt, 'end': r2_retreat_pt, 'orient': 'r2_present_horizontal'},
                action_name="r1_retreat", plan_list=full_plan)
            current_r1_pos = next_pos

        self.interpolate_joint_segment(180,
            r1_args={'start': current_r1_pos, 'end': r1_home_pos, 'start_orient': 'r1_receive_vertical', 'end_orient': 'down'},
            r2_args={'start': r2_retreat_pt, 'end': r2_retreat_pt, 'orient': 'r2_present_horizontal'},
            action_name="r1_retreat", plan_list=full_plan)

        # ====== PHASE 7: SETTLE AND RECORD ======
        last = full_plan[-1]
        for _ in range(120): 
            full_plan.append({"r1": last['r1'], "r2": last['r2'], "action": "r1_retreat"})

        return full_plan