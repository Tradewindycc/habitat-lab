import base64
import json
from turtle import st
import numpy as np
from flask import request
import yaml
import openai
from typing import Any, List, Optional

from .prompts import *

OPENAI_API_KEY = ""


class LLMReflectAgent:
    
    reflect_buffer = []
    def __init__(
        self,
        agent_name: str,
        model="gpt-4o",
        window_size=None,

    ) -> None:
        self.agent_name = agent_name
        self.chat_history = []
        self.window_size = window_size
        self.model = model
        assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY"
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.chat_id = 0

    def chat(self, system_prompt: str = None, content: Any = None, assistant_prompt: str = None, use_history: bool = False, **params):
        content = self._build_content(content) 
        request = self._build_request(system_prompt, content, assistant_prompt, use_history=use_history)
        self._update_chat_history(content)

        response = self.client.chat.completions.create(
            messages=request,  # type: ignore
            model=self.model,
            **self._get_params(params)
        )

        return response.choices[0].message.content

    def _build_content(self, content: Any):

        if isinstance(content, str):
            return content
        elif isinstance(content, np.ndarray):
            base64_image_str = base64.b64encode(content).decode('utf-8')
            image_url = f'data:image/jpeg;base64,{base64_image_str}'
            return image_url

    def _build_request(self, system_prompt: str, content: Any, assistant_prompt: str, use_history: bool = False):
        
        user_prompt = ""
        if use_history:
            user_prompt = f"Our previous talk: {str(self.chat_history)}"
        user_prompt += content

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
        ]
        return messages
    
    def _update_chat_history(self, content):
        self.chat_history.append({"chat_id": self.chat_id, "request": content})
        self.chat_id += 1

    def _get_params(self, params):
        return {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1),
            "frequency_penalty": params.get("frequency_penalty", 0.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
        }

