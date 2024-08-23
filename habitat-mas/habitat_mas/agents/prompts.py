import openai

# TODO(YCC): 1. initial prompt as a assitant. 2.each step prompt: decision maker
system_description = [
"""
You are a helpful assistant.
""",


"""
You are a helpful decision maker and designer in robotics environment. 
"""

]


# TODO(YCC): 1. initial prompt. 2.each step prompt: execute a skill or learn a skill
user_prompt = [
"""
A robot is trying to solve some tasks and learn corresponding skill in the Habitat simulator.

We will provide with you the initial scene description which contains the objects and agents in each episode, we will also provide you with the detailed Robot Resume as agent description.

The scene description example should be in the following format:
``` Scene Description Dictionary
{
    'objects_description': 'There are 1 objects in the scene. 013_apple_:0000 is at position Vector(0.49618, 1.41658, 5.46572).', 
    'agent_description': 'There are 1 agents in the scene. agent_0 is at position [-0.39081627  0.12123175  1.2268406 ].',
    'receptacles_description': 'There are 2 receptacle objects in the scene. And there are 2 navigable receptacle targets on them where object can be placed. frl_apartment_table_01_:0000 contains the following receptacles: ['receptacle_aabb_Tbl1_Top1_frl_apartment_table_01']. frl_apartment_sofa_01_:0000 contains the following receptacles: ['receptacle_aabb_topleft_frl_apartment_sofa_01']. '
}
```

The Robot Resume example should be in the following format:
``` Robot Resume Dictionary
{
    "agent_0": 
        {
            "robot_id": "FetchRobot_default", 
            "robot_type": "FetchRobot", 
            "mobility": {"summary": "The robot has a mobile base with two continuous joints corresponding to left and right wheel links, allowing it to move in a differential drive manner. This enables the robot to navigate across different floors and environments for tasks such as cross-floor object rearrangement and home arrangement."}, 
            "perception": {"summary": "The robot is equipped with a head-mounted camera system that includes RGB and depth cameras, providing it with detailed visual and spatial information about the environment and objects. This camera setup supports tasks requiring cooperative perception and geometric information collection for effective manipulation.", "cameras_info": {"articulated_agent_arm_camera": {"height": 0.79, "type": "articulated"}, "head_camera": {"height": 1.2, "type": "fixed"}}}, 
            "manipulation": {"summary": "The robot has a sophisticated arm with multiple degrees of freedom, including prismatic and revolute joints (shoulder, elbow, wrist, gripper). This allows for complex manipulation tasks such as picking and placing objects, rearranging furniture, and handling objects in challenging positions like high shelves and under tables. The gripper with prismatic joints on both fingers further enhances precise object handling capabilities.", "arm_workspace": {"center": [0.15, 0.9, 0.13], "radius": 1.1, "min_bound": [-0.88, -0.03, -0.89], "max_bound": [1.09, 1.85, 1.1]}}
        }, 
}
```

We also provide the pddl definitions which include the actions you can directly use in the pddl solution and predicates in PDDL.

``` PDDL Definitions yaml
types:
  static_obj_type:
    - art_receptacle_entity_type
    - obj_type
  obj_type:
    - movable_entity_type
    - goal_entity_type
  art_receptacle_entity_type:
    - cab_type
    - fridge_type


constants: {}


predicates:
  - name: in
    args:
      - name: obj
        expr_type: obj_type
      - name: receptacle
        expr_type: art_receptacle_entity_type
    set_state:
      obj_states:
        obj: receptacle

  - name: holding
    args:
      - name: obj
        expr_type: movable_entity_type
      - name: robot_id
        expr_type: robot_entity_type
    set_state:
      robot_states:
        robot_id:
          holding: obj

  - name: not_holding
    args:
      - name: robot_id
        expr_type: robot_entity_type
    set_state:
      robot_states:
        robot_id:
          should_drop: True

  - name: robot_at
    args:
      - name: Y
        expr_type: static_obj_type
      - name: robot_id
        expr_type: robot_entity_type
    set_state:
      robot_states:
        robot_id:
          pos: Y

  - name: robot_at_receptacle
    args:
      - name: receptacle
        expr_type: static_receptacle_entity_type
      - name: robot_id
        expr_type: robot_entity_type
    set_state:
      robot_states:
        robot_id:
          pos: receptacle

  - name: at
    args:
      - name: obj
        expr_type: obj_type
      - name: at_entity
        expr_type: static_obj_type
    set_state:
        obj_states:
            obj: at_entity
  - name: detected_object
    args:
      - name: any_targets|0
        expr_type: movable_entity_type
      - name: robot_id
        expr_type: robot_entity_type
    set_state:
      robot_states:
        robot_id:
          detected_object: any_targets|0

actions:
  - name: nav_to_goal
    parameters:
      - name: obj
        expr_type: movable_entity_type
      - name: robot
        expr_type: robot_entity_type
    precondition:
      # The robot cannot be holding the object that it wants to navigate to.
      expr_type: NAND
      sub_exprs:
        - holding(obj, robot)
    postcondition:
      - robot_at(obj, robot)
    task_info:
      task: NavToObjTask-v0
      task_def: "nav_to_obj"
      config_args:
        task.force_regenerate: True
        task.should_save_to_cache: False

  - name: move_forward
    parameters:
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition: []

  - name: move_backward
    parameters:
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition: []

  - name: turn_left
    parameters:
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition: []

  - name: turn_right
    parameters:
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition: []

  - name: nav_to_obj
    parameters:
      - name: obj
        expr_type: goal_entity_type
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition:
      - robot_at(obj, robot)
    task_info:
      task: NavToObjTask-v0
      task_def: "nav_to_obj"
      config_args:
        task.force_regenerate: True
        task.should_save_to_cache: False

  - name: nav_to_receptacle
    parameters:
      - name: marker
        expr_type: art_receptacle_entity_type
      - name: obj
        expr_type: obj_type
      - name: robot
        expr_type: robot_entity_type
    precondition:
      expr_type: AND
      sub_exprs:
        - in(obj, marker)
    postcondition:
      - robot_at(marker, robot)
    task_info:
      task: NavToObjTask-v0
      task_def: "nav_to_obj"
      config_args:
        task.force_regenerate: True
        task.should_save_to_cache: False

  - name: pick
    parameters:
      - name: obj
        expr_type: movable_entity_type
      - name: robot
        expr_type: robot_entity_type
    precondition:
      expr_type: AND
      sub_exprs:
        - not_holding(robot)
        - robot_at(obj, robot)
        - quantifier: FORALL
          inputs:
            - name: recep
              expr_type: cab_type
          expr_type: NAND
          sub_exprs:
            - in(obj, recep)
            #- closed_cab(recep)
    postcondition:
      - holding(obj, robot)
    task_info:
      task: RearrangePickTask-v0
      task_def: "pick"
      config_args:
        habitat.task.should_enforce_target_within_reach: True
        habitat.task.force_regenerate: True
        habitat.task.base_angle_noise: 0.0
        habitat.task.base_noise: 0.0
        habitat.task.should_save_to_cache: False

  - name: place
    parameters:
      - name: place_obj
        expr_type: movable_entity_type
      - name: obj
        expr_type: goal_entity_type
      - name: robot
        expr_type: robot_entity_type
    precondition:
      expr_type: AND
      sub_exprs:
        - holding(place_obj, robot)
        - robot_at(obj, robot)
    postcondition:
      - not_holding(robot)
      - at(place_obj, obj)
    task_info:
      task: RearrangePlaceTask-v0
      task_def: "place"
      config_args:
        task.should_enforce_target_within_reach: True
        task.force_regenerate: True
        task.base_angle_noise: 0.0
        task.base_noise: 0.0
        task.should_save_to_cache: False

  - name: nav_to_receptacle_by_name
    parameters:
      # - name: marker
      #   expr_type: art_receptacle_entity_type
      - name: receptacle
        expr_type: static_receptacle_entity_type
      - name: robot
        expr_type: robot_entity_type
    precondition: null
    postcondition:
      - robot_at_receptacle(receptacle, robot)
    task_info:
      task: NavToObjTask-v0
      task_def: "nav_to_obj"
      config_args:
        task.force_regenerate: True
        task.should_save_to_cache: False
```


Your goal is listed as following. 1. You need to design a composite task with formated description according to the environment. 2.You need to decompose the task into executable sub-steps as PDDL format for the robot, and for each substep, you should make sure you can either call a skill that already exists in the SKILL_POOL or design a reward to train a new skill in the future.
And your reply must follow the following format as a dictionary:
{
  "task_description": 
  {
    "name": "Rearrange the apple on the table to the sofa",
    "description": "The robot should first navigate to the apple and pick the it from the table, then navigate to the sofa and place the apple on the sofa.",
  },
  "pddl_problem": 
  {
    'objects': [
      {'name': 'any_targets|0', 'expr_type': 'movable_entity_type'}, 
      {'name': 'TARGET_any_targets|0', 'expr_type': 'goal_entity_type'}, 
      {'name': 'robot_0', 'expr_type': 'robot_entity_type'}
    ], 
    'goal': {
      'expr_type': 'AND', 
      'sub_exprs': ['at(any_targets|0,TARGET_any_targets|0)', 'not_holding(robot_0)']
    }, 
    'stage_goals': {
      'stage_1': {'expr_type': 'AND', 'sub_exprs': ['robot_at(any_targets|0, robot_0)']}, 
      'stage_2': {'expr_type': 'AND', 'sub_exprs': ['holding(any_targets|0, robot_0)']}, 
      'stage_3': {'expr_type': 'AND', 'sub_exprs': ['at(any_targets|0, TARGET_any_targets|0)']}
    }, 
    'solution': ['nav_to_goal(any_targets|0, robot_0)', 'pick(any_targets|0, robot_0)', 'nav_to_obj(TARGET_any_targets|0, robot_0)', 'place(any_targets|0,TARGET_any_targets|0, robot_0)']
  },
}
Another example reply:
{
  "task_description":
  {
    "name": "get the apple and find the sofa",
    "description": "The robot should navigate to the apple and pick it, then navigate to the sofa.",
  },
  "pddl_problem":
  {
    'objects': [
      {'name': 'any_targets|0', 'expr_type': 'movable_entity_type'},
      {'name': 'frl_apartment_chair_01_:0000|receptacle_aabb_Chr1_Top1_frl_apartment_chair_01', 'expr_type': 'static_receptacle_entity_type'},
      {'name': 'robot_0', 'expr_type': 'robot_entity_type'}
    ],
    'goal': {
      'expr_type': 'AND',
      'sub_exprs': ['robot_at_receptacle(frl_apartment_chair_01_:0000|receptacle_aabb_Chr1_Top1_frl_apartment_chair_01, robot_0)', 'holding(any_targets|0, robot_0)']
    },
    'stage_goals': {
      'stage_1': {'expr_type': 'AND', 'sub_exprs': ['robot_at(any_targets|0, robot_0)']},
      'stage_2': {'expr_type': 'AND', 'sub_exprs': ['holding(any_targets|0, robot_0)']},
      'stage_3': {'expr_type': 'AND', 'sub_exprs': ['robot_at_receptacle(frl_apartment_chair_01_:0000|receptacle_aabb_Chr1_Top1_frl_apartment_chair_01, robot_0)']}
    },
    'solution': ['nav_to_goal(any_targets|0, robot_0)', 'pick(any_targets|0, robot_0)', 'nav_to_receptacle_by_name(frl_apartment_chair_01_:0000|receptacle_aabb_Chr1_Top1_frl_apartment_chair_01, robot_0)']
  },
}


In the pddl_problem yaml presented above, any_targets|0 represent the apple, TARGET_any_targets|0 represent the sofa, robot_0 represent the robot, the solution is the sequence of sub-steps, each of which is an executable skill or a skill to learn. Remember the object(like apple here) names should always be any_targets|0, any_targets|1, etc. when goal_entity_type(sofa here) name should be TARGET_any_targets|0 etc. Robot name should be robot_0. Receptacle name should be mentioned in the receptacle description.
Please make sure the dictionary you generate will be strictly follow the dcit format which can be load into json and remember to add , after }. 

The scene description and robot resume are listed below, please generate the task description and decomposed PDDL as formatted above.

""",



# TODO(YCC): give the example of observation for each step and reward design
"""
The robot is solving the tasks right now in a step. 
Choose to execute the skill in PDDL solution to achieve the stage goal, or design a reward function to learn the new skill. 
We will provide the environment observation from all the sensors on the robot for this step.

The obervation example should be in the following format:
``` Observation Dictionary
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
```
Also, we will provide previous skills you have chosen to execute and their execution results. 
If you choose to directly execute the skill in PDDL solution, you should simply reply with a dictionary to execute the skill for this step as you set in PDDL like this:
{
  "step_choice": "execution",
  "action_name": "nav_to_obj",
}
If you choose to design a reward function to learn the new skill in the solution, then you should design a reward function with the success measurement, special task inherited from SetArticulatedObjectTask, training config.
SetArticulatedObjectTask is defined as the base class for all tasks involving manipulating articulated objects:
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
        
{
  "step_choice": "learn",
  "action_name": "close_fridge",
}
``` Reward Design


```
``` Skill task 
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
```
``` Training config
# @package _global_
defaults:
  - /habitat: habitat_config_base
  - /habitat/simulator: rearrange_sim
  - /habitat/simulator/agents/habitat_mas_agents@habitat.simulator.agents.agent_0: FetchRobot_default
  - /habitat/task: task_config_base
  - /habitat/task/actions@habitat.task.actions.arm_action: oracle_arm_action
  - /habitat/task/actions@habitat.task.actions.base_velocity: base_velocity_non_cylinder
  - /habitat/task/actions@habitat.task.actions.rearrange_stop: rearrange_stop
  - /habitat/task/actions@habitat.task.actions.pddl_apply_action: pddl_apply_action
  - /habitat/task/actions@habitat.task.actions.oracle_nav_action: fetch_oracle_nav_action
  - /habitat/task/measurements:
    - articulated_agent_force
    - force_terminate
    - articulated_agent_colls
    - ee_dist_to_marker
    - end_effector_to_rest_distance
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
    type: RearrangeCloseFridgeTask-v0
    reward_measure: art_obj_reward
    success_measure: art_obj_success
    success_reward: 10.0
    slack_reward: 0.0
    end_on_success: True
    use_marker_t: False
    actions:
      arm_action:
        disable_grip: True
    measurements:
      art_obj_reward:
        wrong_grasp_end: True
        ee_dist_reward: 1.0
        marker_dist_reward: 1.0
  environment:
    max_episode_steps: 200
  dataset:
    data_path: data/datasets/manipulation/manipulation_eval.json.gz
```

""",
]

assistant_prompt = [
"""
Sure, I understand my goal. The task description and PDDL is as follows:
""",

"""
According to the observation in this step, my choice to achieve the stage goal is as follows:
"""
]
