from habitat_mas.scene_graph import scene_description
import openai

# TODO(YCC): 1. initial prompt as a assitant. 2.each step prompt: decision maker
system_prompts = [
"""
You are a helpful assistant.
""",

"""
You are a helpful assistant expected to design and decompose a task in the Habitat simulator.
""",

"""
You are a success verifier to indicate whether the robot subgoal is satisfied given the success measurements.
"""

]


# TODO(YCC): 1. initial prompt. 2.each step prompt: execute a skill or learn a skill
user_prompts = [
# 0. prompt with environment context and pddl definitions
"""
A robot is trying to solve some tasks and learn corresponding skill in the Habitat simulator.
We will provide with you initial scene description which contains three parts: 
1.Scene description contains objects and agents in this episode.
2.Robot resume as robot description.
3.PDDL definitions which includes the actions you can directly use in the PDDL solution.

The scene description is in the following format:[SCENE DESCRIPTION]. For articulated receptacles, the robot can interact with the markers on them; for rigid receptacles, the robot can only pick or place objects on them.
The robot resume is in the following format:[ROBOT RESUME]
The PDDL definitions is in the following format:[PDDL DEFINITIONS]

Now can you specify the robots, objects, receptacles and actions that robot can take in the environment as a dictionary?
""",

# 1. specify the goal of llm
"""
Good job! Now your jobs are: 
1.Design a composite task , try to decompose it into sub-steps and describe them. All the elements in your designed task must be included in the scene description. You should notice that some of substeps' action may not be in the PDDL definition, which means you should learn a new skill for that.
2.Decide whether to learn a new skill according to task description and PDDL definition. If not, go to step 3.
3.Decompose the task into executable sub-steps as PDDL format for the robot.

If you choose to learn a new skill according to your designed task, your reply format can refer to: [TASK DICT YES]
If you choose to directly execute a PDDL solution according to your designed task, your reply format can refer to: [TASK DICT NO]

In the pddl_problem yaml presented above, any_targets|0 represent the object, TARGET_any_targets|0 represent a goal entity which could be a receptacle position, robot_0 represent the robot, the solution is the sequence of sub-steps, which are defined in the pddl definition.Remember that all the action used in PDDL solution must be included in the pddl definitions.
Please make sure the dictionary you generate will be strictly follow the dcit format which can be load into json and remember to add , after }. 

Now please design a novel task according to the scene description, try to decompose it and make your decision on whether to learn a new skill or execute the PDDL solution.
""",



# give the example of observation for each step and reward design
"""
The robot is solving the tasks right now in a step. 
Choose to execute the skill in PDDL solution to achieve the stage goal, or design a reward function to learn the new skill. 
We will provide the environment observation from all the sensors on the robot for this step.

The obervation example should be in the following format:
[OBSERVATION DICTIONARY]

Also, we will provide previous skills you have chosen to execute and their execution results. 
If you choose to directly execute the skill in PDDL solution, you should simply reply with a dictionary to execute the skill for this step as you set in PDDL like this:
[STEP EXECUTION]
If you choose to design a reward function to learn a new skill: First, you should reply with [STEP LEARN], then you should design 1. a reward function with the success measurement, 2. special task inherited from SetArticulatedObjectTask(SetArticulatedObjectTask is defined as the base class for all tasks involving manipulating articulated objects:[BASE TASK]), 3. training config.
1. reward function with the success measurement will be like [REWARD EXAMPLE]
2. task example used to train the new skill will be like [TASK EXAMPLE]
3. training config example used to config the training will be like [TRAINING CONFIG]

""",
]

assistant_prompts = [
"""
{'robots': [{'robot_name': '...', 'robot_type': '...', 'mobility': '...', 'perception': '...', 'manipulation': '...'}], 'objects': [{'object_id': '...', 'object_type': '...', 'position': [...]}, ...], 'receptacles': [{'receptacle_id': '...', 'receptacle_type': '...', 'target_places': [...]}, ...], 'actions': [{'action_name': '...', 'parameters': [...]}, ...}]}
""",

"""
{"choice": "...", "task_description": {"name": "...", "description": "..."}, "pddl_problem": {...}, "new_skills": [...]}
"""
]

task_dict = [
"""{"choice": "YES", "task_description": {"name": "Get the [OBJECT] in the fridge onto the [GOAL_RECEPTACLE].", "description": "1.navigate to the fridge 2.open the fridge 3.pick the [OBJECT] in fridge 4.navigate to the [GOAL_RECEPTACLE] 5.place the [OBJECT] on the [GOAL_RECEPTACLE]."}, "pddl_problem": {}, "new_skills": [{"action_name": "open_fridge", "parameters": ["fridge", "robot"]}]}""" ,   
"""{"choice": "NO", "task_description": {"name": "Rearrange the [OBJECT] on the [TARGET_RECEPTACLE] to the [GOAL_RECEPTACLE]", "description": "1.navigate to the [OBJECT] 2.pick the [OBJECT] from the [TARGET_RECEPTACLE] 3.navigate to the [GOAL_RECEPTACLE] 4.place the [OBJECT] on the [GOAL_RECEPTACLE]."}, "pddl_problem": {'objects': [{'name': 'any_targets|0', 'expr_type': 'movable_entity_type'}, {'name': 'TARGET_any_targets|0', 'expr_type': 'goal_entity_type'}, {'name': 'robot_0', 'expr_type': 'robot_entity_type'}], 'goal': {'expr_type': 'AND', 'sub_exprs': ['at(any_targets|0,TARGET_any_targets|0)', 'not_holding(robot_0)']}, 'stage_goals': {'stage_1': {'expr_type': 'AND', 'sub_exprs': ['robot_at(any_targets|0, robot_0)']}, 'stage_2': {'expr_type': 'AND', 'sub_exprs': ['holding(any_targets|0, robot_0)']}, 'stage_3': {'expr_type': 'AND', 'sub_exprs': ['at(any_targets|0, TARGET_any_targets|0)']}}, 'solution': ['nav_to_goal(any_targets|0, robot_0)', 'pick(any_targets|0, robot_0)', 'nav_to_obj(TARGET_any_targets|0, robot_0)', 'place(any_targets|0,TARGET_any_targets|0, robot_0)']}, "new_skills": []}""",

]

observation_dict = """
{
  'all_predicates': tensor([[0., 0., 0., 1., 0., 0.]], device='cuda:0'), 
  'arm_workspace_rgb': tensor([[[[255, 255, 255],...,]], device='cuda:0', dtype=torch.uint8), 
  'articulated_agent_arm_depth': tensor([[[[0.8244],...,]]], device='cuda:0'), 
  'articulated_agent_arm_rgb': tensor([[[[138, 111,  55],...,]]], device='cuda:0', dtype=torch.uint8), 
  'articulated_agent_arm_semantic': tensor([[[[  3],...,]]], device='cuda:0', dtype=torch.int32), 
  'detected_objects': tensor([[  0,   3, 103, 108, 110, 112, 118]], device='cuda:0',dtype=torch.int32), 
  'ee_pos': tensor([[ 0.6306, -0.2504,  0.9977]], device='cuda:0'), 'has_finished_oracle_nav': tensor([[0.]], device='cuda:0'), 
  'head_depth': tensor([[[[2.4797],...,]]], device='cuda:0'), 
  'head_rgb': tensor([[[[255, 255, 255],...,]]], device='cuda:0', dtype=torch.uint8), 
  'head_semantic': tensor([[[[3],...,]]], device='cuda:0', dtype=torch.int32), 
  'is_holding': tensor([[0.]], device='cuda:0'), 
  'joint': tensor([[-4.5000e-01, -1.0800e+00,  1.0000e-01,  9.3500e-01, -1.0000e-03,
          1.5730e+00,  5.0000e-03]], device='cuda:0'), 
  'localization_sensor': tensor([[-0.3908,  0.1212,  1.2268,  1.3145]], device='cuda:0'), 
  'obj_goal_gps_compass': tensor([[5.8456, 2.0123]], device='cuda:0'), 
  'obj_goal_sensor': tensor([[-0.2140, -5.8281, -1.0942]], device='cuda:0'), 
  'obj_start_gps_compass': tensor([[4.3307, 2.6790]], device='cuda:0'), 
  'obj_start_sensor': tensor([[-0.9790, -3.1652, -3.4995]], device='cuda:0'), 
  'relative_resting_position': tensor([[-0.1306,  0.2504,  0.0023]], device='cuda:0'), 
  'third_rgb': tensor([[[[163, 158, 147],...,]]], device='cuda:0', dtype=torch.uint8)
}
"""

step_reply = [
"""
{
  "step_choice": "execution",
  "action_name": "nav_to_obj",
}
""",
"""
{
  "step_choice": "learn",
  "action_name": "open_fridge",
}
"""

]

task_example = [
"""
@registry.register_task(name="RearrangeCloseFridgeTask-v0")
class RearrangeCloseFridgeTaskV1(SetArticulatedObjectTask):
    def _get_spawn_region(self):
        return mn.Range2D([0.833, -0.6], [1.25, 0.6])

    def _get_look_pos(self):
        return [0.0, 0.0, 0.0]

    def _gen_start_state(self):
        return np.array([0, np.random.uniform(np.pi / 4, 2 * np.pi / 3)])

    def reset(self, episode: Episode):
        self._use_marker = "fridge_push_point"
        return super().reset(episode)
""",
]

training_config = """
# @package _global_

defaults:
  - /habitat: habitat_config_base

  - /habitat/simulator: rearrange_sim
  - /habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent: depth_head_agent
  - /habitat/simulator/agents@habitat.simulator.agents.main_agent: fetch_suction

  - /habitat/task: task_config_base
  - /habitat/task/rearrange/actions: fetch_suction_arm_base_stop
  - /habitat/task/measurements:
    - articulated_agent_force
    - force_terminate
    - articulated_agent_colls
    - end_effector_to_rest_distance
    - ee_dist_to_marker
    - art_obj_at_desired_state
    - art_obj_state
    - does_want_terminate
    - art_obj_success
    - art_obj_reward
    - num_steps
    - did_violate_hold_constraint
  - /habitat/task/lab_sensors:
    - joint_sensor
    - is_holding_sensor
    - end_effector_sensor
    - relative_resting_pos_sensor

  - /habitat/dataset/rearrangement: replica_cad
  - _self_

habitat:
  gym:
    obs_keys:
      - head_depth
      - joint
      - ee_pos
      - is_holding
      - relative_resting_position
  task:
    type: RearrangeOpenFridgeTask-v0
    reward_measure: art_obj_reward
    success_measure: art_obj_success
    success_reward: 10.0
    slack_reward: 0.0
    end_on_success: True
    use_marker_t: False
    success_state: 1.2207963268
    actions:
      arm_action:
        grip_controller: SuctionGraspAction
    measurements:
      art_obj_at_desired_state:
        use_absolute_distance: false
      art_obj_reward:
        type: ArtObjReward
        wrong_grasp_end: True
        grasp_reward: 5.0
        marker_dist_reward: 1.0
        ee_dist_reward: 1.0
        constraint_violate_pen: 1.0
      force_terminate:
        max_accum_force: 20_000.0
        max_instant_force: 10_000.0
  environment:
    max_episode_steps: 200
  dataset:
    data_path: data/datasets/open_fridge/in_fridge_1k.json.gz
"""

base_task = """
class SetArticulatedObjectTask(RearrangeTask):
    # Base class for all tasks involving manipulating articulated objects.
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self._use_marker: str = None
        self._prev_awake = True
        self._force_use_marker = None
    @property
    def use_marker_name(self) -> str:
        # The name of the target marker the agent interacts with.
        return self._use_marker
    def get_use_marker(self) -> MarkerInfo:
        # The marker the agent should interact with.
        return self._sim.get_marker(self._use_marker)
    def set_args(self, marker, obj, **kwargs):
        if marker.startswith("MARKER_"):
            marker = marker[len("MARKER_") :]
        self._force_use_marker = marker
        # The object in the container we are trying to reach and using as the
        # position of the container.
        self._targ_idx = obj
    @property
    def success_js_state(self) -> float:
        # The success state of the articulated object desired joint.
        return self._config.success_state
    @abstractmethod
    def _gen_start_state(self) -> np.ndarray:
        pass
    @abstractmethod
    def _get_look_pos(self) -> np.ndarray:
        # The point defining where the robot should face at the start of the episode.
    @abstractmethod
    def _get_spawn_region(self) -> mn.Range2D:
        # The region on the ground the robot can be placed.
    def _sample_robot_start(self, T) -> Tuple[float, np.ndarray]:
        # Returns the start face direction and the starting position of the robot.
        spawn_region = self._get_spawn_region()
        if self._config.spawn_region_scale == 0.0:
            # No randomness in the base position spawn
            start_pos = spawn_region.center()
        else:
            spawn_region = mn.Range2D.from_center(
                spawn_region.center(),
                self._config.spawn_region_scale * spawn_region.size() / 2,
            )
            start_pos = np.random.uniform(spawn_region.min, spawn_region.max)
        start_pos = np.array([start_pos[0], 0.0, start_pos[1]])
        targ_pos = np.array(self._get_look_pos())
        # Transform to global coordinates
        start_pos = np.array(T.transform_point(mn.Vector3(*start_pos)))
        start_pos = np.array([start_pos[0], 0, start_pos[2]])
        start_pos = self._sim.safe_snap_point(start_pos)
        targ_pos = np.array(T.transform_point(mn.Vector3(*targ_pos)))
        # Spawn the robot facing the look pos
        forward = np.array([1.0, 0, 0])
        rel_targ = targ_pos - start_pos
        angle_to_obj = get_angle(forward[[0, 2]], rel_targ[[0, 2]])
        if np.cross(forward[[0, 2]], rel_targ[[0, 2]]) > 0:
            angle_to_obj *= -1.0
        return angle_to_obj, start_pos
    def step(self, action: Dict[str, Any], episode: Episode):
        return super().step(action, episode)
    @property
    def _is_there_spawn_noise(self):
        return (
            self._config.base_angle_noise != 0.0
            or self._config.spawn_region_scale != 0.0
        )
    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        if self._force_use_marker is not None:
            self._use_marker = self._force_use_marker
        marker = self.get_use_marker()
        if self._config.use_marker_t:
            T = marker.get_current_transform()
        else:
            ao = marker.ao_parent
            T = ao.transformation
        jms = marker.ao_parent.get_joint_motor_settings(marker.joint_idx)
        if self._config.joint_max_impulse > 0:
            jms.velocity_target = 0.0
            jms.max_impulse = self._config.joint_max_impulse
        marker.ao_parent.update_joint_motor(marker.joint_idx, jms)
        num_timeout = 100
        self._disable_art_sleep()
        for _ in range(num_timeout):
            self._set_link_state(self._gen_start_state())
            angle_to_obj, base_pos = self._sample_robot_start(T)
            noise = np.random.normal(0.0, self._config.base_angle_noise)
            self._sim.articulated_agent.base_rot = angle_to_obj + noise
            self._sim.articulated_agent.base_pos = base_pos
            articulated_agent_T = (
                self._sim.articulated_agent.base_transformation
            )
            rel_targ_pos = articulated_agent_T.inverted().transform_point(
                marker.current_transform.translation
            )
            if not self._is_there_spawn_noise:
                rearrange_logger.debug(
                    "No spawn noise, returning first found position"
                )
                break
            eps = 1e-2
            upper_bound = (
                self._sim.articulated_agent.params.ee_constraint[0, :, 1] + eps
            )
            is_within_bounds = (rel_targ_pos < upper_bound).all()
            if not is_within_bounds:
                continue
            did_collide = False
            for _ in range(self._config.settle_steps):
                self._sim.internal_step(-1)
                did_collide, details = rearrange_collision(
                    self._sim,
                    self._config.count_obj_collisions,
                    ignore_base=False,
                )
                if did_collide:
                    break
            if not did_collide:
                break
        # Step so the updated art position evaluates
        self._sim.internal_step(-1)
        self._reset_art_sleep()
        self.prev_dist_to_push = -1
        self.prev_snapped_marker_name = None
        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)
    def _disable_art_sleep(self):
        # Disables the sleeping state of the articulated object. Use when setting
        ao = self.get_use_marker().ao_parent
        self._prev_awake = ao.awake
        ao.awake = True
    def _reset_art_sleep(self) -> None:
        # Resets the sleeping state of the target articulated object.
        ao = self.get_use_marker().ao_parent
        ao.awake = self._prev_awake
    def _set_link_state(self, art_pos: np.ndarray) -> None:
        # Set the joint state of all the joints on the target articulated object.
        ao = self.get_use_marker().ao_parent
        ao.joint_positions = art_pos
"""