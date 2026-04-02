import pybullet as p
import numpy as np

class SinglePandaController:
    def __init__(self, robot_id, cube_ids):
        self.robot_id = robot_id
        self.cube_ids = cube_ids
        
        self.arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [9, 10]
        self.ee_link_index = 11  
        
        self.reset_to_ready_pose()

    def reset_to_ready_pose(self):
        ready_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for i, joint_idx in enumerate(self.arm_indices):
            p.resetJointState(self.robot_id, joint_idx, ready_q[i])
            
    def get_object_pos(self, name):
        if name in self.cube_ids:
            pos, _ = p.getBasePositionAndOrientation(self.cube_ids[name])
            return np.array(pos)
        return None

    def get_base_pose(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos)
        
    def get_base_rotation(self):
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        return np.array(rot_matrix).reshape(3,3)

    def set_commands(self, joints, gripper_state):
        p.setJointMotorControlArray(
            self.robot_id, self.arm_indices, 
            p.POSITION_CONTROL, targetPositions=joints,
            forces=[240]*7 
        )
        
        # FIXED GRIPPER VALUES:
        # Open: 0.04 (8cm total gap - clearance for 7cm cube)
        # Closed: 0.005 (1cm total gap - tight grip on 7cm cube) 
        finger_target = 0.04 if gripper_state > 0.03 else 0.005
        
        p.setJointMotorControlArray(
            self.robot_id, self.gripper_indices, 
            p.POSITION_CONTROL, targetPositions=[finger_target, finger_target],
            forces=[40, 40] 
        )
    def is_gripper_fully_open(self):
        """Reads the ACTUAL physical position of the simulation motors"""
        joint_states = p.getJointStates(self.robot_id, self.gripper_indices)
        left_finger = joint_states[0][0]
        right_finger = joint_states[1][0]
        
        # 0.04 is maximum open. If they are >0.035, they have safely cleared the box.
        return (left_finger > 0.038) and (right_finger > 0.038)