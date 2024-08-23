# ruff: noqa
from typing import Any, Dict, List, Tuple

import torch

from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_mas.agents.actions.arm_actions import *
from habitat_mas.agents.actions.base_actions import *
from habitat_mas.agents.llm_reflection_agent import LLMReflectAgent
from habitat_mas.agents.prompts import *
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData

ACTION_POOL = [get_agents, send_request, nav_to_obj, nav_to_goal, pick, place, nav_to_receptacle_by_name]

class LLMHighLevelReflectPolicy(HighLevelPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._active_envs = torch.zeros(self._num_envs, dtype=torch.bool)

        llm_actions = ACTION_POOL
        # Initialize the LLM agent
        self._llm_agent = self._init_llm_agent(kwargs["agent_name"], llm_actions)
        self.step_mode = "pddl"

        self._update_solution_actions(
            [self._parse_solution_actions() for _ in range(self._num_envs)]
        )

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def _setup_step_mode(self, step_mode):
        self.step_mode = step_mode

    def _init_llm_agent(self, agent_name, action_list):
        return LLMReflectAgent(
            agent_name=agent_name,
            system_prompt=system_description[0],
            action_space=action_list,
        )
    
    # TODO(YCC): initialize the LLM agent with environment config and reset pddl problem
    def reset_pddl(self, content):
        self._llm_agent = self._init_llm_agent(self._agent_name, ACTION_POOL)
        task_description, pddl_problem = self._llm_agent.chat(
            content = str(content),
            user_prompt = user_prompt[0], 
            assistant_prompt = assistant_prompt[0],
            mode = "initial"
            )
        # TODO(YCC): make sure the pddl_problem is valid for yaml
        return task_description, pddl_problem

    def _parse_function_call_args(self, llm_args: Dict) -> str:
        """
        Parse the arguments of a function call from the LLM agent to the policy input argument format.
        """
        return llm_args

    def apply_mask(self, mask):
        """
        Apply the given mask to the agent in parallel envs.

        Args:
            mask: Binary mask of shape (num_envs, ) to be applied to the agents.
        """
        self._active_envs = mask
        self._next_sol_idxs *= mask.cpu().view(-1)

    # TODO(YCC): get the next skill by different mode
    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor, PolicyActionData]:
        """
        Get the next skill to execute from the LLM agent.
        """
        # TODO: use these text context to query the LLM agent with function call
        envs_text_context = kwargs.get("envs_text_context", None)
        agent_task_assignments = kwargs.get("agent_task_assignments", None)
        
        batch_size = masks.shape[0]
        next_skill = torch.zeros(batch_size)
        skill_args_data = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            if self.step_mode == 'llm':
            # Query the LLM agent with the current observations
            # to get the next action and arguments
                llm_output = self._llm_agent.chat(content=str(observations[batch_idx]), user_prompt=user_prompt[1], mode="llm_step")
                if llm_output is None:
                    next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                    skill_args_data[batch_idx] = ["1"]
                    continue

                action_name = llm_output["name"]
                action_args = self._parse_function_call_args(llm_output["arguments"])

                if action_name in self._skill_name_to_idx:
                    next_skill[batch_idx] = self._skill_name_to_idx[action_name]
                    skill_args_data[batch_idx] = action_args
                else:
                    # If the action is not valid, do nothing
                    next_skill[batch_idx] = self._skill_name_to_idx["wait"]
                    skill_args_data[batch_idx] = ["1"]

            elif self.step_mode == 'pddl':
                llm_output = self._llm_agent.chat(content=str(observations[batch_idx]), user_prompt=user_prompt[1], assistant_prompt=assistant_prompt[1], mode="pddl_step")
                use_idx = self._get_next_sol_idx(batch_idx, immediate_end)

                skill_name, skill_args = self._solution_actions[batch_idx][
                    use_idx
                ]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args  # type: ignore[call-overload]

                self._next_sol_idxs[batch_idx] += 1
            else:
                raise ValueError(f"Invalid step mode: {self.step_mode}, select step mode between llm and pddl.")               

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(),
        )
    
    def _update_solution_actions(
        self, solution_actions: List[List[Tuple[str, List[str]]]]
    ) -> None:
        if len(solution_actions) == 0:
            raise ValueError(
                "Solution actions must be non-empty (if want to execute no actions, just include a no-op)"
            )
        self._solution_actions = solution_actions

    def _parse_solution_actions(self) -> List[Tuple[str, List[str]]]:
        """
        Returns the sequence of actions to execute as a list of:
        - The action name.
        - A list of the action arguments.
        """
        solution = self._pddl_prob.solution

        solution_actions = []
        for i, hl_action in enumerate(solution):
            sol_action = (
                hl_action.name,
                [x.name for x in hl_action.param_values],
            )
            # Avoid adding plan actions that are assigned to other agents to the list.
            agent_idx = self._agent_name.split("_")[-1]
            for j, param in enumerate(hl_action.params):
                param_name = param.name
                param_value = hl_action.param_values[j].name
                # if the action is assigned to current agent, add it to the list
                if param_name == "robot" and param_value.split("_")[-1] == agent_idx:
                    solution_actions.append(sol_action)
            
            if self._config.add_arm_rest and i < (len(solution) - 1):
                solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        solution_actions.append(parse_func("wait(3000)"))

        return solution_actions
    
    def _get_next_sol_idx(self, batch_idx, immediate_end):
        """
        Get the next index to be used from the list of solution actions.

        Args:
            batch_idx: The index of the current environment.

        Returns:
            The next index to be used from the list of solution actions.
        """
        if self._next_sol_idxs[batch_idx] >= len(
            self._solution_actions[batch_idx]
        ):
            baselines_logger.info(
                f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
            )
            immediate_end[batch_idx] = True
            # Just repeat the last action.
            return len(self._solution_actions[batch_idx]) - 1
        else:
            return self._next_sol_idxs[batch_idx].item()

    def filter_envs(self, curr_envs_to_keep_active):
        """
        Cleans up stateful variables of the policy so that
        they match with the active environments
        """
        self._next_sol_idxs = self._next_sol_idxs[curr_envs_to_keep_active]
        parse_solution_actions = [
            self._parse_solution_actions() for _ in range(self._num_envs)
        ]
        self._update_solution_actions(
            [
                parse_solution_actions[i]
                for i in range(curr_envs_to_keep_active.shape[0])
                if curr_envs_to_keep_active[i]
            ]
        )
    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )