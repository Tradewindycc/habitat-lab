# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import magnum as mn
from magnum import Vector3
import numpy as np
from gym import spaces
import habitat.utils.visualizations.maps as maps
import habitat_sim
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction,
    BaseVelNonCylinderAction,
    HumanoidJointAction,
)
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.tasks.utils import get_angle
from habitat_sim.physics import VelocityControl
from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavAction,
    OracleNavCoordinateAction
)

DEBUG_SAVE_NAVMESH = False
DEBUG_SAVE_PATHFINDER_MAP = False

@registry.register_task_action
class OracleNavDiffBaseAction(OracleNavAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    
    Compared with OracleNavAction, this action will use different navmesh for different agent bases. 
    
    """

    def __init__(self, *args, **kwargs):
        config = kwargs["config"]
        task = kwargs["task"]
        self.motion_type = config.motion_control
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "base_velocity_non_cylinder":
            BaseVelNonCylinderAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav  action")

        self._task = task
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )
        self._prev_ep_id = None
        self.skill_done = False
        self._targets = {}
        self.config = config
        self.pathfinder = None
        self.step_sum = 0

    def _create_pathfinder(self, config):
        """
        Create a pathfinder for the current agent base
        """
        pf = habitat_sim.nav.PathFinder()
        # deepcopy does not work since NavMeshSettings is non-pickleable
        # modified_settings = deepcopy(self._sim.pathfinder.nav_mesh_settings)
        modified_settings = self._sim.pathfinder.nav_mesh_settings
        # modified_settings = habitat_sim.NavMeshSettings()
        # modified_settings.set_defaults()
        # for key in dir(self._sim.pathfinder.nav_mesh_settings):
        #     attr = getattr(self._sim.pathfinder.nav_mesh_settings, key)
        #     if not key.startswith("__") and not callable(attr):
        #         setattr(
        #             modified_settings,
        #             key,
        #             attr
        #         )
                
        modified_settings.agent_radius = config.agent_radius
        modified_settings.agent_height = config.agent_height
        modified_settings.agent_max_climb = config.agent_max_climb
        modified_settings.agent_max_slope = config.agent_max_slope
        modified_settings.include_static_objects = True
        # Create a new pathfinder with slightly stricter radius to provide nav buffer from collision
        modified_settings.agent_radius += 0.05
        assert self._sim.recompute_navmesh(
            pf, modified_settings
        ), "failed to recompute navmesh"
        
        # assert self._sim.recompute_navmesh(
        #     pf, self._sim.pathfinder.nav_mesh_settings
        # ), "failed to recompute navmesh"
        
        # DEBUG: save recomputed navmesh
        if DEBUG_SAVE_NAVMESH:
            from habitat_mas.perception.nav_mesh import NavMesh
            import open3d as o3d
            
            if self._agent_index == 0:
                navmesh_vertices = self._sim.pathfinder.build_navmesh_vertices()
                navmesh_indices = self._sim.pathfinder.build_navmesh_vertex_indices()
                nav_mesh = NavMesh(
                    vertices=np.stack(navmesh_vertices, axis=0),
                    triangles=np.array(navmesh_indices).reshape(-1, 3),
                )
                o3d.io.write_triangle_mesh("navmesh_default.ply", nav_mesh.mesh)
            
            
            navmesh_vertices = pf.build_navmesh_vertices()
            navmesh_indices = pf.build_navmesh_vertex_indices()
            nav_mesh = NavMesh(
                vertices=np.stack(navmesh_vertices, axis=0),
                triangles=np.array(navmesh_indices).reshape(-1, 3),
            )
            o3d.io.write_triangle_mesh(f"navmesh_agent_{self._agent_index}.ply", nav_mesh.mesh)
            self._sim.pathfinder = pf
            

            
        return pf

    def _plot_map_and_path(self, kwargs, pathfinder: habitat_sim.nav.PathFinder, 
                           save_name="sim", map_resolution=1024):
        """
        Plot the top-down map and the path on it
        """
        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            return
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        # base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ, pathfinder)

        # Create a top-down map using the pathfinder and a specified height
        topdown_map = maps.get_topdown_map(
            pathfinder=pathfinder,
            height=0.1522,
            map_resolution=map_resolution,
        )
        # Colorize the top-down map for visualization
        colorize_topdown_map = maps.colorize_topdown_map(topdown_map)
        # Convert the current path points to 2D coordinates
        curr_path_points_2d = [((pt[2]), (pt[0])) for pt in curr_path_points]

        # Convert the current path points to grid coordinates in the top-down map
        # for i in range(0, len(curr_path_points_2d)):
        #     x, y = maps.to_grid(realworld_x=curr_path_points_2d[i][1], realworld_y=curr_path_points_2d[i][0], grid_resolution=[7295, 4096], pathfinder=pathfinder)
        #     curr_path_points_2d[i] = [x, y]
        grid_resolution = topdown_map.shape
        for i in range(0, len(curr_path_points_2d)):
            x, y = maps.to_grid(
                realworld_x=curr_path_points_2d[i][0],
                realworld_y=curr_path_points_2d[i][1],
                grid_resolution=grid_resolution,
                pathfinder=pathfinder,
            )
            curr_path_points_2d[i] = [x, y]

        # Draw the path on the colorized top-down map
        maps.draw_path(colorize_topdown_map, curr_path_points_2d)

        # Get the robot's position and rotation
        base_pos = np.array(self.cur_articulated_agent.base_pos)
        x, y = maps.to_grid(
            realworld_x=base_pos[2],
            realworld_y=base_pos[0],
            grid_resolution=grid_resolution,
            pathfinder=pathfinder,
        )
        robot_rot = np.array(self.cur_articulated_agent.base_rot)
        # Draw the agent's position and rotation on the top-down map
        maps.draw_agent(image=colorize_topdown_map, agent_center_coord=[x, y], 
                        agent_rotation=robot_rot, agent_radius_px=30)

        return colorize_topdown_map

    def step_filter(self, start_pos: Vector3, end_pos: Vector3) -> Vector3:
        r"""Computes a valid navigable end point given a target translation on the NavMesh.
        Uses the configured sliding flag.

        :param start_pos: The valid initial position of a translation.
        :param end_pos: The target end position of a translation.
        """
        if self.pathfinder.is_loaded:
            if self.config.allow_dyn_slide:
                end_pos = self.pathfinder.try_step(start_pos, end_pos)
            else:
                end_pos = self.pathfinder.try_step_no_sliding(start_pos, end_pos)

        return end_pos

    def update_base(self, *args, **kwargs):
        """
        Choose the update_base method based on the motion type
        """
        if self.motion_type == "base_velocity":
            return BaseVelAction.update_base(self, *args, **kwargs)
        elif self.motion_type == "base_velocity_non_cylinder":
            return BaseVelNonCylinderAction.update_base(self, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized motion type {self.motion_type} for update_base function")

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (
                np.array(start_pos),
                np.array(obj_pos),
            )
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point, pathfinder=None):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        :param pathfinder: The pathfinder to use for computing the path
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        if pathfinder is None:
            found_path = self.pathfinder.find_path(path)
        else:
            found_path = pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, **kwargs):
        # if pathfinder is not created, create it
        if not self.pathfinder:
            self.pathfinder = self._create_pathfinder(self.config)

        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            return
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        # DEBUG: plot the map and path
        if DEBUG_SAVE_PATHFINDER_MAP:
            sim_map = self._plot_map_and_path(kwargs, self._sim.pathfinder, save_name="sim")
            agent_map = self._plot_map_and_path(kwargs, self.pathfinder, save_name="agent")
            
            # Get episode id
            ep_id = self._sim.ep_info.episode_id

            if not os.path.exists(f'./debug/{ep_id}'):
                os.makedirs(f'./debug/{ep_id}')

            # Display the colorized top-down map
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)

            # Plot sim_map
            axs[0].imshow(sim_map)
            axs[0].set_title(f'Sim Map')
            axs[0].axis('on')

            # Plot agent_map
            axs[1].imshow(agent_map)
            axs[1].set_title(f'Agent Map')
            axs[1].axis('on')

            # Save the figure
            plt.savefig(
                os.path.join(
                    f"./debug/{ep_id}",
                    f"{self._action_arg_prefix}_{self.step_sum}"
                    + ".png",
                ),
                bbox_inches="tight",
                pad_inches=0,
            )
            # plt.close(fig)
                
            self.step_sum += 1

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                BaseVelAction.step(self, *args, **kwargs)
                return

            elif self.motion_type == "base_velocity_non_cylinder":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                BaseVelNonCylinderAction.step(self, *args, **kwargs)
                return

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True

                base_action = self.humanoid_controller.get_pose()
                kwargs[f"{self._action_arg_prefix}human_joints_trans"] = (
                    base_action
                )

                HumanoidJointAction.step(self, *args, **kwargs)
                return
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


@registry.register_task_action
class OracleNavCoordinateAction(BaseVelAction, BaseVelNonCylinderAction, HumanoidJointAction):  # type: ignore
    """
    An action to drive the anget to a given coordinate
    """

    def __init__(self, *args, **kwargs):
        config = kwargs["config"]
        task = kwargs["task"]
        self.motion_type = config.motion_control
        self._targets = {}
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)
            

        elif self.motion_type == "base_velocity_non_cylinder":
            BaseVelNonCylinderAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav  action")

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def update_base(self, *args, **kwargs):
        """
        Choose the update_base method based on the motion type
        """
        if self.motion_type == "base_velocity":
            return BaseVelAction.update_base(self, *args, **kwargs)
        elif self.motion_type == "base_velocity_non_cylinder":
            return BaseVelNonCylinderAction.update_base(self, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized motion type {self.motion_type} for update_base function")

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self._sim.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_coord_action": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                self._action_arg_prefix
                + "oracle_nav_lookat_action": spaces.Box(
                    shape=(3,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                ),
                self._action_arg_prefix + "mode": spaces.Discrete(3),
            }
        )

    def _get_target_for_coord(self, look_at_pos, agent_pos, mode=0):
        """Given a place to look at, selects an agent_pos to navigate"""
        if mode == 0:
            return (agent_pos, np.array(look_at_pos))
        if mode == 2:
            return (agent_pos, np.array(agent_pos))
        precision = 0.25
        pos_key = np.around(look_at_pos / precision, decimals=0) * precision
        pos_key = tuple(pos_key)

        if pos_key not in self._targets:
            agent_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(look_at_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            self._targets[pos_key] = agent_pos
        else:
            agent_pos = self._targets[pos_key]
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (agent_pos, np.array(look_at_pos))

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, **kwargs):
        # mode = 0,1,2
        # 0 means both target and lookat are given, 1 means only lookat is given, 2 means only target is given
        self.skill_done = False
        nav_to_target_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_lookat_action",
        )
        nav_position_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_coord_action",
        )
        mode = kwargs.get(self._action_arg_prefix + "mode")
        if np.linalg.norm(nav_to_target_coord) == 0:
            return {}

        final_nav_targ, obj_targ_pos = self._get_target_for_coord(
            nav_to_target_coord, nav_position_coord, mode
        )

        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)
        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )

            # Distance at which we don't need to check angle
            # this is because in some cases we may be very close to the object
            # which causes instability in the angle_to_obj
            distance_close_no_distance = self._config.dist_thresh / 10.0
            at_goal = dist_to_final_nav_targ < distance_close_no_distance or (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos,
                            self._config.turn_velocity,
                            robot_forward,
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ,
                            self._config.turn_velocity,
                            robot_forward,
                        )
                else:
                    vel = [0, 0]
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(self, *args, **kwargs)

            elif self.motion_type == "base_velocity_non_cylinder":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction._compute_turn(
                            rel_pos,
                            self._config.turn_velocity,
                            robot_forward,
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction._compute_turn(
                            rel_targ,
                            self._config.turn_velocity,
                            robot_forward,
                        )
                else:
                    vel = [0, 0]
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelNonCylinderAction.step(self, *args, **kwargs)

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        if self._config["lin_speed"] == 0:
                            distance_multiplier = 0.0
                        else:
                            distance_multiplier = 1.0
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                            distance_multiplier,
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True
                # This line is important to reset the controller
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[f"{self._action_arg_prefix}human_joints_trans"] = (
                    base_action
                )

                return HumanoidJointAction.step(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )


class SimpleVelocityControlEnv:
    """
    Simple velocity control environment for moving agent
    """

    def __init__(self, sim_freq=120.0):
        # the velocity control
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._sim_freq = sim_freq

    def act(self, trans, vel):
        linear_velocity = vel[0]
        angular_velocity = vel[1]
        # Map velocity actions
        self.vel_control.linear_velocity = mn.Vector3(
            [linear_velocity, 0.0, 0.0]
        )
        self.vel_control.angular_velocity = mn.Vector3(
            [0.0, angular_velocity, 0.0]
        )
        # Compute the rigid state
        rigid_state = habitat_sim.RigidState(
            mn.Quaternion.from_matrix(trans.rotation()), trans.translation
        )
        # Get the target rigit state based on the simulation frequency
        target_rigid_state = self.vel_control.integrate_transform(
            1 / self._sim_freq, rigid_state
        )
        # Get the ending pos of the agent
        end_pos = target_rigid_state.translation
        # Offset the height
        end_pos[1] = trans.translation[1]
        # Construct the target trans
        target_trans = mn.Matrix4.from_(
            target_rigid_state.rotation.to_matrix(),
            target_rigid_state.translation,
        )

        return target_trans
