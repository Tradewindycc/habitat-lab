import json
import yaml
import re
import openai
from typing import List, Optional

from .crab_core import Action

from .prompts import *

OPENAI_API_KEY = ""


class LLMReflectAgent:
    
    reflect_buffer = []
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        action_space: List[Action],
        model="gpt-3.5-turbo",
        window_size=None,

    ) -> None:
        self.agent_name = agent_name
        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }
        self._convert_action_to_schema(action_space)
        self.action_map = {action.name: action for action in action_space}
        self.chat_history = []
        self.window_size = window_size
        self.model = model
        self.initialized = False
        self.env_context = None
        self.pddl_task = None
        assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY"
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )

    def chat(self, content: str, user_prompt: str = "", assistant_prompt: str = "", mode: str = "step"):
        user_message = {
            "role": "user",
            "content": user_prompt + " " + content
        }
        assitant_message = {
            "role": "assistant",
            "content": assistant_prompt
        }
        request = [self.system_message, user_message, assitant_message]

        if self.window_size is None:
            for message in self.chat_history:
                request = request + message
        elif self.window_size > 0 and len(self.chat_history) > 0:
            for message in self.chat_history[-self.window_size :]:
                request = request + message
        
        self.chat_history.append([user_message, assitant_message])

        # TODO(YCC): return response in different mode
        if mode == "initial":
            # TODO(YCC): prompt needs include the actions information
            self.env_context = content
            request[1]['content'] += ("Actions in the pool: " + str(self.actions))
            response = self.client.chat.completions.create(
                messages=request,  # type: ignore
                model=self.model,
                temperature=0.7,
                # max_tokens=2048,
            )

            response_message = response.choices[0].message.content
            response_message = re.sub(r'\s+', ' ', response_message).strip()
            response_message = response_message.replace("'", '"')
            response_message = re.sub(r'(?<!\\)""', '"', response_message)
            response_dict = json.loads(response_message)
            self.initialized = True
            self.pddl_task = response_dict['pddl_problem']
            return response_dict['task_description'], response_dict['pddl_problem']
        
        # in the mode of llm_step, llm directly choose a action in the action space to act and return the parameters
        elif mode == "llm_step":
            response = self.client.chat.completions.create(
                messages=request,  # type: ignore
                model=self.model,
                tools=[{"type": "function", "function": action} for action in self.actions],
                tool_choice="required",
            )

            response_message = response.choices[0].message
            self.chat_history[-1].append(response_message)

            tool_calls = response_message.tool_calls
            for tool_call in tool_calls:
                self.chat_history[-1].append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": "",
                    }
                )  # extend conversation with function response

            call = tool_calls[0]
            parameters = json.loads(call.function.arguments)
            return {"name": call.function.name, "arguments": parameters}
        
        # TODO(YCC): act as pddl solution, if not in pddl solution, return reward
        elif mode == "pddl_step":
            response = self.client.chat.completions.create(
                messages=request,  # type: ignore
                model=self.model,
                temperature=0.7,
            )
            response_message = response.choices[0].message.content
            response_message = re.sub(r'\s+', ' ', response_message).strip()
            response_message = response_message.replace("'", '"')
            response_message = re.sub(r'(?<!\\)""', '"', response_message)
            response_dict = json.loads(response_message)
            return response_dict

    def _convert_action_to_schema(self, action_space):
        self.actions = []
        for action in action_space:
            new_action = action.to_openai_json_schema()
            self.actions.append(new_action)
