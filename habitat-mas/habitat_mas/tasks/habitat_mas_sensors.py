import os
from typing import List, Dict, Any
import numpy as np
import json
from gym import spaces
from dataclasses import dataclass, field

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat_mas.dataset.defaults import habitat_mas_data_dir
from habitat_mas.scene_graph.scene_graph_hssd import SceneGraphHSSD
from habitat_mas.scene_graph.utils import generate_objects_description, generate_agents_description, gernerate_receptacles_description    

@registry.register_sensor
class HSSDSceneDescriptionSensor(Sensor):
    """Sensor to generate text descriptions of the scene from the environment simulation."""
    # TODO: consider if all scene description sensors should have the same uuid, since only one can be used in a dataset
    cls_uuid: str = "scene_description"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)
        
    def _get_uuid(self, *args, **kwargs):
        return HSSDSceneDescriptionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def get_observation(self, *args, **kwargs):
        """Generate text descriptions of the scene."""
        
        # Initialize scene graph
        sg = SceneGraphHSSD()
        sg.load_gt_scene_graph(self._sim)
        
        # Generate scene descriptions
        objects_description = generate_objects_description(sg.object_layer)
        agent_description = generate_agents_description(sg.agent_layer, sg.region_layer, sg.nav_mesh)
        receptacle_description = gernerate_receptacles_description(sg.sim)
        
        scene_description = {
            "objects_description": objects_description,
            "agent_description": agent_description,
            "receptacles_description": receptacle_description
        }
        
        # convert dict to json string
        scene_description_str = json.dumps(scene_description)
        
        return scene_description_str
    
@registry.register_sensor
class RobotResumeSensor(Sensor):
    """Sensor to load and retrieve robot resumes from JSON files."""
    cls_uuid: str = "robot_resume"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self.robot_resume_dir = os.path.join(habitat_mas_data_dir, 
                                             config.robot_resume_dir)
        self.robot_resumes = self.load_robot_resumes()
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs):
        return RobotResumeSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def get_observation(self, robot_configs, *args, **kwargs):
        """
        Retrieve the resumes for all agents as a single string.
        """
            
        robot_resumes = {}
        for agent_config in robot_configs:
            # agent_handle = agent_config["agent_type"]
            agent_handle = f"agent_{agent_config['agent_idx']}"
            agent_type = agent_config["agent_type"]
            robot_resume_file = os.path.join(self.robot_resume_dir, f"{agent_type}.json")
            if os.path.exists(robot_resume_file):
                with open(robot_resume_file, "r") as f:
                    robot_resume = json.load(f, parse_float=lambda x: round(float(x), 2))
                    robot_resumes[agent_handle] = robot_resume
        
        # convert dict to json string
        robot_resumes_str = json.dumps(robot_resumes)
        
        return robot_resumes_str

    def load_robot_resumes(self) -> Dict[str, Dict]:
        """
        Load robot resumes from JSON files located in the robot_resume_dir directory.
        The method reads the agent handles from the environment config and loads their corresponding resumes.
        """
        robot_resumes = {}
        for file in os.listdir(self.robot_resume_dir):
            if file.endswith(".json"):
                agent_handle = file.split(".")[0]
                robot_resume_file = os.path.join(self.robot_resume_dir, file)
                with open(robot_resume_file, "r") as f:
                    robot_resume = json.load(f)
                    robot_resumes[agent_handle] = robot_resume

        return robot_resumes

def get_text_sensors(sim, *args, **kwargs):
    
    # sensor_keys = kwargs.get("sensor_keys", None)
    # sensor_keys = kwargs.get("sensor_keys", {})
    lab_sensors_config = kwargs.get("lab_sensors_config", None)
    
    
    if lab_sensors_config is None:
        lab_sensors_config = {}
    
        @dataclass
        class HSSDSceneDescriptionSensorConfig:
            type: str = "HSSDSceneDescriptionSensor"

        @dataclass
        class RobotResumeSensorConfig:
            type: str = "RobotResumeSensor"
            robot_resume_dir: str = "robot_resume"
            
        lab_sensors_config["scene_description"] = HSSDSceneDescriptionSensorConfig()
        lab_sensors_config["robot_resume"] = RobotResumeSensorConfig()
        
    return {
        "scene_description": HSSDSceneDescriptionSensor(
            sim, lab_sensors_config["scene_description"], *args, **kwargs
        ),
        "robot_resume": RobotResumeSensor(
            sim, lab_sensors_config["robot_resume"], *args, **kwargs
        )
    }

def get_text_context(sim, robot_configs: List[Dict], *args, **kwargs):
    
    text_sensors = get_text_sensors(sim, *args, **kwargs)
    text_context = {
        sensor_name: text_sensor.get_observation(
            robot_configs=robot_configs
        ) 
        for sensor_name, text_sensor in text_sensors.items()
    }
    return text_context

if __name__ == "__main__":
    env_config = {
        "agent_handles": ["FetchRobot", "SpotRobot"]
    }
    robot_resume_dir = "../data/robot_resume"
    robot_resume_sensor = RobotResumeSensor(env_config, robot_resume_dir)

    fetch_robot_resume = robot_resume_sensor.get_robot_resume("FetchRobot")
    print(fetch_robot_resume)
    
    spot_robot_resume = robot_resume_sensor.get_robot_resume("SpotRobot")
    print(spot_robot_resume)
    