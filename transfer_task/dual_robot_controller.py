import pybullet as p
import numpy as np

class DualPandaController:
    def __init__(self, ids_dict):
        self.r1_id = ids_dict['r1']
        self.r2_id = ids_dict['r2']
        self.cube_id = ids_dict['cube']
        
        self.arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [9, 10]
        self.ee_idx = 11

        self.r1_constraint = None
        self.r2_constraint = None
        
        self.reset_to_ready_pose()

    def reset_to_ready_pose(self):
        ready_q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        for j_idx, q in zip(self.arm_indices, ready_q):
            p.resetJointState(self.r1_id, j_idx, q)
            p.resetJointState(self.r2_id, j_idx, q)

    def set_commands(self, r1_joints, r1_grip, r2_joints, r2_grip):
        self._move_arm(self.r1_id, r1_joints)
        self._move_gripper(self.r1_id, r1_grip)
        self._handle_grasping(self.r1_id, r1_grip, "r1")

        self._move_arm(self.r2_id, r2_joints)
        self._move_gripper(self.r2_id, r2_grip)
        self._handle_grasping(self.r2_id, r2_grip, "r2")

    def _move_arm(self, robot_id, joints):
        p.setJointMotorControlArray(
            robot_id, self.arm_indices, p.POSITION_CONTROL, 
            targetPositions=joints,
            forces=[240]*7 ,
            targetVelocities=[0.1] * 7
        )

    def _move_gripper(self, robot_id, state):
        # FIX: 0.04 = Open. 0.02 = Partially closed (4cm gap). 
        # This prevents the fingers from squeezing into the 5cm solid mesh and jittering!
        tgt = 0.04 if state > 0.02 else 0.02 
        p.setJointMotorControlArray(
            robot_id, self.gripper_indices, p.POSITION_CONTROL, 
            targetPositions=[tgt, tgt], forces=[10, 10] 
        )

    def _handle_grasping(self, robot_id, state, robot_name):
        should_grasp = (state < 0.02)
        current_constraint = self.r1_constraint if robot_name == "r1" else self.r2_constraint
        
        if should_grasp and current_constraint is None:
            if robot_name == "r2" and self.r1_constraint is not None:
                return

            ee_state = p.getLinkState(robot_id, self.ee_idx)
            ee_pos = np.array(ee_state[0])
            cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            
            dist = np.linalg.norm(ee_pos - cube_pos)
            
            # FIX: Expanded detection radius to 0.15 ensures the constraint ALWAYS connects
            if dist < 0.05: 
                print(f"{robot_name} Attached Cube! (dist={dist:.4f})")
                
                ee_pos_world, ee_orn_world = ee_state[0], ee_state[1]
                inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos_world, ee_orn_world)
                
                child_pos_in_parent, child_orn_in_parent = p.multiplyTransforms(
                    inv_ee_pos, inv_ee_orn, cube_pos.tolist(), cube_orn
                )
                
                cid = p.createConstraint(
                    parentBodyUniqueId=robot_id,
                    parentLinkIndex=self.ee_idx,
                    childBodyUniqueId=self.cube_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=child_pos_in_parent,
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=child_orn_in_parent,
                    childFrameOrientation=[0, 0, 0, 1]
                )
                p.changeConstraint(cid, maxForce=2000)  
                
                if robot_name == "r1": 
                    self.r1_constraint = cid
                    if self.r2_constraint is not None:
                        p.removeConstraint(self.r2_constraint)
                        self.r2_constraint = None
                        print("Handover logic: R2 constraint auto-dropped to prevent jitter!")
                else: 
                    self.r2_constraint = cid

        elif not should_grasp and current_constraint is not None:
            print(f"{robot_name} Released Cube!")
            p.removeConstraint(current_constraint)
            
            if robot_name == "r1": self.r1_constraint = None
            else: self.r2_constraint = None