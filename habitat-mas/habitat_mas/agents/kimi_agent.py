import random
from typing import *

import os
import json
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from openai import OpenAI


# from habitat_mas.agents.llm_agent_base import LLMAgentBase


class KimiAgent:
    def __init__(self, **kwargs):
        self.agent_index = kwargs.get("agent_index", 0)
        with open('api.json', 'r') as file:
            api = json.load(file)
        self.key = api['kimi']['key']
        self.client = OpenAI(
            base_url="https://api.moonshot.cn/v1",
            api_key=self.key,
        )
        self.message = []
        self.system_msg = {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，"
                       "准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不"
                       "可翻译成其他语言。",
        }
        self.history = [self.system_msg]

    def chat(self, **kwargs) -> Optional[dict]:
        transform = transforms.ToPILImage()
        sensor_list = ['arm_workspace_rgb', 'head_rgb', 'head_semantic']
        image_paths = [f'kimi_images/{sensor}.png' for sensor in sensor_list]
        for sensor in sensor_list:
            if sensor in kwargs['observation']:
                if len(kwargs['observation'][sensor].size()) >= 3:
                    image = transform(
                        kwargs['observation'][sensor].permute(2, 0, 1))
                    image.save(f'kimi_images/{sensor}.png')

        # file_messages = self._upload_images(images=image_paths)
        query = {
                "role": "user",
                "content": f"""你的目标是: {kwargs['pddl_prob'].goal};\n你的阶段性目标是: {kwargs['pddl_prob'].stage_goals};\n你的标准规划方式是: {kwargs['pddl_prob'].solution};\n你目前所属的状态是{kwargs['pddl_prob'].predicates}\n\n请你规划下一步的行为, 你可以从以下操作中选择: {kwargs['actions']}""",
            }
        self.history.append(query)
        multi_modal_messages = [
            # *file_messages,
            self.system_msg,
            query
        ]
        completion = self.client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=multi_modal_messages,
            temperature=0.3,
        )
        result = completion.choices[0].message.content
        self.history.append({"role": "assistant", "content": result})
        query = """请你以 dict 格式输出你的动作，如:{
                "name": "</action_name>",
                "arguments": {
                    </pddl_entity_name>: </pddl_entity_name>,
                    ...
                    }
                }""" + f"你的</action_name>即之前你所选择的在{kwargs['skill_names']}的动作，你可以选择的合适的</pddl_entity_name>填入，你的pddl_entity_name选择有{kwargs['pddl_prob'].all_entities}"
        self.history.append({"role": "user", "content": query})
        completion = self.client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=self.history,
            temperature=0.3,
        )
        result = completion.choices[0].message.content

        return result

    def _upload_images(self, images: List[str]) -> List[Dict[str, Any]]:
        messages = []
        for image in images:
            file_object = self.client.files.create(file=Path(image), purpose="file-extract")
            file_content = self.client.files.content(file_id=file_object.id).text
            messages.append({
                "role": "system",
                "content": file_content,
            })

        return messages
