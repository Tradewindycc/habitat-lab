from typing import List, Optional

from .crab_core import Action
from ..utils.models import OpenAIModel


REQUEST_TEMPLATE = '"{source_agent}" agent sent you requests: "{request}".'

ROBOT_EXECUTION_SYSTEM_PROMPT_TEMPLATE = (
    'You are a "{robot_type}" agent called "{robot_key}".'
    " Your task is to work with other agents to complete the task described below:\n\n"
    '"""\n{task_description}\n"""\n\n'
    "You MUST take finish subtask assgined to you:"
    '"""\n{subtask_description}\n"""\n\n'
    # "You have the following capabilities:\n\n"
    # '"""\n{capabilities}\n"""\n\n'
    "You MUST take one and only one action using function call in each step."
    " If you think the task definitely cannot be done by yourself, you can use `send_request` function to ask other agents for help."
)


class CrabAgent:
    message_pipe: dict[str, list[str]] = {}

    def __init__(
        self,
        name: str,
        actions: List[Action],
    ):
        self.name = name
        self.llm_model = OpenAIModel("", action_space=actions)
        self.initilized = False

    def init_agent(
        self,
        robot_type: str,
        task_description: str,
        subtask_description: str,
    ):
        """This function is a hack to intilize agent after the object is created"""
        self.robot_type = robot_type
        self.task_description = task_description
        self.subtask_description = subtask_description
        self.llm_model.set_system_message(
            ROBOT_EXECUTION_SYSTEM_PROMPT_TEMPLATE.format(
                robot_type=self.robot_type,
                robot_key=self.name,
                task_description=task_description,
                subtask_description=subtask_description,
            )
        )
        self.initilized = True

    def chat(self, observation: str) -> Optional[dict]:
        if self.name in CrabAgent.message_pipe and CrabAgent.message_pipe[self.name]:
            prompt = " ".join(CrabAgent.message_pipe[self.name])
            observation = str(observation) + " " + prompt
            CrabAgent.message_pipe[self.name] = []

        action_name, parameters = self.llm_model.chat(str(observation))
        if action_name == "send_request":
            target_agent = parameters["target_agent"]
            if target_agent == self.name:  # send request to itself
                return None
            request = parameters["request"]
            if target_agent not in CrabAgent.message_pipe:
                CrabAgent.message_pipe[target_agent] = []
            prompt = REQUEST_TEMPLATE.format(source_agent=self.name, request=request)
            CrabAgent.message_pipe[target_agent].append(prompt)
            return {"name": "wait", "arguments": ["20"]}
        if action_name == "wait":
            return {"name": "wait", "arguments": ["20"]}
        if action_name in ["nav_to_obj", "nav_to_goal", "nav_to_robot", "place", "pick"]:
            parameters["robot"] = self.name
            return {"name": action_name, "arguments": parameters}
        else:
            return {"name": action_name, "arguments": parameters}
