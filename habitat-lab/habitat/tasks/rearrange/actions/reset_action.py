from typing import Optional, Union
import magnum
import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction,
    ArmEEAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import MotionType


@registry.register_task_action
class OracleResetArmAction(ArmEEAction, ArticulatedAgentAction):
    """
    Pick/drop action for the articulated_agent given the object/receptacle index.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        ArmEEAction.__init__(self, *args, sim=sim, **kwargs)
        self._sim = sim
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self._prev_ep_id = None
        self._hand_pose_iter = 0
        self.is_init = False
        self.is_reset = False

    def reset(self, *args, **kwargs):
        super().reset()

    @property
    def action_space(self):
        return spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )

    def set_desired_ee_pos(self, ee_pos: np.ndarray) -> None:
        self.ee_target += np.array(ee_pos)

        self.apply_ee_constraints()

        joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
            self.cur_articulated_agent.arm_motor_pos = des_joint_pos
        if self.cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
            self.cur_articulated_agent.arm_joint_pos = des_joint_pos
            self.cur_articulated_agent.fix_joint_values = des_joint_pos

    def step(self, pick_action, **kwargs):
        should_reset = pick_action[0]
        if not self.is_init:
            self._init_joint_pos = np.array(self.cur_articulated_agent.arm_joint_pos)
            self.is_init = True

        if should_reset == 3:
            # or self.cur_grasp_mgr.snap_idx is None
            # TODO(zxz): even if is settled initially, the two pos are still not matched
            # 1) cur ee_pos is not right
            # 2) calc_fk is wrong
            # 两个问题：cur_articulated_agent里面的pos和初始的pos不匹配
            # cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
            cur_ee_pos = self._ik_helper.calc_fk(self.cur_articulated_agent.arm_joint_pos)
            if not self.is_reset:
                self.ee_target = cur_ee_pos
                self.is_reset = True
            init_ee_pos = self._ik_helper.calc_fk(self._init_joint_pos)
            translation = magnum.Vector3(init_ee_pos) - cur_ee_pos

            # translation from object to end effector in base frame
            # translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)
            translation_base = np.clip(translation, -1, 1)
            self._ee_ctrl_lim = 0.03
            translation_base *= self._ee_ctrl_lim
            self.set_desired_ee_pos(translation_base)

            # DEBUG VISUALIZATION
            if self._render_ee_target:
                global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
                self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                    global_pos, self._sim.viz_ids["ee_target"]
                )
        else:
            self.is_reset = False

        return self.ee_target
