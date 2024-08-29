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
        action_space: List[Action],
        model="gpt-3.5-turbo",
        window_size=None,

    ) -> None:
        self.agent_name = agent_name
        self._convert_action_to_schema(action_space)
        self.action_map = {action.name: action for action in action_space}
        self.chat_history = []
        self.window_size = window_size
        self.model = model
        self.initialized = False
        self.pddl_task = None
        assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY"
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.chat_id = 0

    def chat(self, system_prompt: str = "", user_prompt: str = "", assistant_prompt: str = "", **params):
        previous_talk = f"Previou request: {str(self.chat_history)}\n"

        system_message = {
            "role": "system",
            "content": system_prompt,
        }
        user_message = {
            "role": "user",
            "content": previous_talk + user_prompt
        }
        assitant_message = {
            "role": "assistant",
            "content": assistant_prompt
        }
        request = [system_message, user_message, assitant_message]

        self.chat_history.append({"chat_id": self.chat_id,"request": user_prompt})
        self.chat_id += 1

        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1)
        frequency_penalty = params.get("frequency_penalty", 0.0)
        presence_penalty = params.get("presence_penalty", 0.0)

        response = self.client.chat.completions.create(
            messages=request,  # type: ignore
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        response_message = response.choices[0].message.content


        return response_message  

    def _convert_action_to_schema(self, action_space):
        self.actions = []
        for action in action_space:
            new_action = action.to_openai_json_schema()
            self.actions.append(new_action)
