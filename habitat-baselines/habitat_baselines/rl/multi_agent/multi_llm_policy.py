from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import gym.spaces as spaces
import numpy as np
import torch

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)

from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiPolicy,
    MultiAgentPolicyActionData,
    MultiStorage,
    MultiUpdater,
    _merge_list_dict
)

class MultiLLMPolicy(MultiPolicy):
    """
    Wraps a set of LLM policies. Add group discussion stage before individual policy actions.
    """

    def __init__(self, update_obs_with_agent_prefix_fn):
        self._active_policies = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

    def set_active(self, active_policies):
        self._active_policies = active_policies

    def on_envs_pause(self, envs_to_pause):
        for policy in self._active_policies:
            policy.on_envs_pause(envs_to_pause)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        envs_text_context=[{}],
        **kwargs,
    ):

        n_agents = len(self._active_policies)
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1] * n_agents, -1)
        n_envs = prev_actions.shape[0]

        # Stage 1: If all prev_actions are zero, which means it is the first step of the episode, then we need to do group discussion
        # Given: Robot resume + Scene description + task instruction
        # Output: (Subtask decomposition) + task assignment
        envs_task_assignments = []
        for i in range(n_envs):
            env_prev_actions = prev_actions[i]
            env_text_context = envs_text_context[i]
            # if no previous actions, then it is the first step of the episode
            if not env_prev_actions.any():
                if "robot_resume" in env_text_context:
                    robot_resume = env_text_context["robot_resume"]
                if "scene_description" in env_text_context:
                    scene_description = env_text_context["scene_description"]
                #TODO: Add group discussion here
                # task_assignments = group_discussion(robot_resume, scene_description)
                # {
                #     "agent_0": "Look for the object xx in the environment",
                #     "agent_1": "Navigate to object xxx, and pick up it, placing it at receptacle yyy",
                #     ...
                # }
                envs_task_assignments.append({})
                pass


        # Stage 2: Individual policy actions
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            # collect assigned tasks for agent_i across all envs
            agent_i_handle = f"agent_{agent_i}"
            agent_task_assignments = [task_assignment[agent_i]
                                      if agent_i_handle in task_assignment else ""
                                      for task_assignment in envs_task_assignments]

            agent_obs = self._update_obs_with_agent_prefix_fn(
                observations, agent_i
            )

            agent_actions.append(
                policy.act(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                    deterministic,
                    envs_text_context=envs_text_context,
                    agent_task_assignments=agent_task_assignments # pass the task planning result to the policy
                )
            )
        policy_info = _merge_list_dict(
            [ac.policy_info for ac in agent_actions]
        )
        batch_size = masks.shape[0]
        device = masks.device

        action_dims = split_index_dict["index_len_prev_actions"]

        # We need to split the `take_actions` if they are being assigned from
        # `actions`. This will be the case if `take_actions` hasn't been
        # assigned, like in a monolithic policy where there is no policy
        # hierarchicy.
        if any(ac.take_actions is None for ac in agent_actions):
            length_take_actions = action_dims
        else:
            length_take_actions = None

        def _maybe_cat(get_dat, feature_dims, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros(
                    (batch_size, feature_dims[ind]), device=device, dtype=dtype
                )
                if dat is None
                else dat
                for ind, dat in enumerate(all_dat)
            ]
            return torch.cat(all_dat, -1)

        rnn_hidden_lengths = [
            ac.rnn_hidden_states.shape[-1] for ac in agent_actions
        ]
        return MultiAgentPolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(
                lambda ac: ac.actions, action_dims, prev_actions.dtype
            ),
            values=_maybe_cat(
                lambda ac: ac.values, [1] * len(agent_actions), torch.float32
            ),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs,
                [1] * len(agent_actions),
                torch.float32,
            ),
            take_actions=torch.cat(
                [
                    ac.take_actions
                    if ac.take_actions is not None
                    else ac.actions
                    for ac in agent_actions
                ],
                -1,
            ),
            policy_info=policy_info,
            should_inserts=np.concatenate(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else np.ones(
                        (batch_size, 1), dtype=bool
                    )  # None for monolithic policy, the buffer should be updated
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            length_take_actions=length_take_actions,
            num_agents=n_agents,
        )

    def _build_index_split(self, rnn_hidden_states, prev_actions, kwargs):
        """
        Return a dictionary with rnn_hidden_states lengths and action lengths that
        will be used to split these tensors into different agents. If the lengths
        are already in kwargs, we return them as is, if not, we assume agents
        have the same action/hidden dimension, so the tensors will be split equally.
        Therefore, the lists become [dimension_tensor // num_agents] * num_agents
        """
        n_agents = len(self._active_policies)
        index_names = [
            "index_len_recurrent_hidden_states",
            "index_len_prev_actions",
        ]
        split_index_dict = {}
        for name_index in index_names:
            if name_index not in kwargs:
                if name_index == "index_len_recurrent_hidden_states":
                    all_dim = rnn_hidden_states.shape[-1]
                else:
                    all_dim = prev_actions.shape[-1]
                split_indices = int(all_dim / n_agents)
                split_indices = [split_indices] * n_agents
            else:
                split_indices = kwargs[name_index]
            split_index_dict[name_index] = split_indices
        return split_index_dict

    def get_value(
        self, observations, rnn_hidden_states, prev_actions, masks, **kwargs
    ):
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = torch.split(
            rnn_hidden_states,
            split_index_dict["index_len_recurrent_hidden_states"],
            dim=-1,
        )
        agent_prev_actions = torch.split(
            prev_actions, split_index_dict["index_len_prev_actions"], dim=-1
        )
        agent_masks = torch.split(masks, [1, 1], dim=-1)
        all_value = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(
                observations, agent_i
            )
            all_value.append(
                policy.get_value(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                )
            )
        return torch.stack(all_value, -1)

    def get_extra(
        self, action_data: MultiAgentPolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        all_extra = []
        for policy in self._active_policies:
            all_extra.append(policy.get_extra(action_data, infos, dones))
        # The action_data is shared across all policies, so no need to reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        all_discrete = np.all(
            [
                isinstance(policy.policy_action_space, spaces.MultiDiscrete)
                for policy in self._active_policies
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [
                        policy.policy_action_space.n
                        for policy in self._active_policies
                    ]
                )
            )
        else:
            return spaces.Dict(
                {
                    policy_i: policy.policy_action_space
                    for policy_i, policy in enumerate(self._active_policies)
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for policy in self._active_policies:
            if isinstance(policy.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(policy.policy_action_space, spaces.Box):
                lens.append(policy.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {policy.policy_action_space}"
                    "not supported."
                )
        return lens

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn)


class MultiLLMStorage(MultiStorage):
    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        super().__init__(update_obs_with_agent_prefix_fn, **kwargs)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class MultiLLMUpdater(MultiUpdater):
    def __init__(self):
        self._active_updaters = []

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiLLMUpdater()
