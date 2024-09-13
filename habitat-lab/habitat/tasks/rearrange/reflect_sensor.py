import cv2
import json
import habitat
import quaternion
import habitat_sim
import numpy as np
import magnum as mn
import os.path as osp
from gym import spaces
from dataclasses import dataclass
from matplotlib import pyplot as plt
from typing import Any, List, Optional, Tuple, Dict

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat_mas.tasks.habitat_mas_sensors import RobotResumeSensor
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles, AABBReceptacle
from habitat.config.default_structured_configs import RobotResumeSensorConfig, EnvStartTextSensorConfig


space_map = {
    "rigid": "on the surface of",
    "articulated": "inside or on the surface of"
}

motion_type_map = {
    "kinematic": (
        "they are controlled by the system and does not respond to collisions or other physical events. "
        "They can only be picked up from receptacles and placed to other receptacles.\n"
    ),
    "dynamic": (
        "they are affected by external forces and adheres to physical simulation rules. "
        "They can respond to collisions and other physical events. "
        "They can be picked up, placed or even thrown, allowing for a range of interactions.\n"
    )
}

articulated_map = {
    "rigid": (
        "these receptacles have no movable parts. "
        "Objects can be placed on them, "
        "but they cannot be manipulated or interact with any joints. \n"
    ),
    "articulated": (
        "these receptacles have movable parts or joints that can be interacted with. "
        "Objects can be placed inside or on them, "
        "and their articulated joints or markers can be manipulated. \n"
    )
}

OBJECT_DESCRIPTION = (
    "{name} is an {type} object with the PDDL label '{label}', "
    "and is currently placed {space_relation} the receptacle '{parent_recep}' "
    "at the {parent_aabb_name} bounding box. \n"
)


RIGID_RECEPTACLE_DESCRIPTION = (
    "{name} is a rigid receptacle, meaning it has no movable parts. "
    "It contains {aabbs} . "
    "Currently, {child_objects} are placed on it. "
    "This receptacle is associated with the PDDL target labels '{labels}', "
    "which typically refer to itself. \n"
)

ARTICULATED_RECEPTACLE_DESCRIPTION = (
    "{name} is an articulated receptacle. "
    "It has {aabbs} AABBReceptacles where objects can be placed. "
    "Currently, {child_objects} are placed in or on it. "
    "There are {markers} contact points on this receptacle where you can interact with. "
    "This receptacle is associated with the PDDL target labels '{labels}', "
    "which could refer to itself.\n"
)

def find_aabb_by_receptacle_handle(sim, receptacle_handle):
    all_aabbreceptacles = find_receptacles(sim)

    aabb = []

    for aabb_receptacle in all_aabbreceptacles:
        parent_handle = aabb_receptacle.parent_object_handle
        
        if parent_handle == receptacle_handle:
            aabb.append(aabb_receptacle)
    return aabb

def find_markers_by_receptacle_handle(sim, receptacle_handle, art_type = 'rigid'):
    if art_type == 'rigid':
        return []

    markers = sim.ep_info.markers
    markers_on_recep = []

    for marker in markers:
        assert "params" in marker and "object" in marker["params"]
        if marker["params"]["object"] == receptacle_handle:
            markers_on_recep.append(marker)
    return markers_on_recep


class SceneObject:
    def __init__(
            self,   
            name: str, 
            origin:Any = None,
            sim_id:int = None, 
            pos:Any = None, 
            label:str = None, 
            parent_receptacle = None,
            parent_aabb: AABBReceptacle = None,
            motion_type: habitat_sim.physics.MotionType = None
    ):
        self.name = name
        self.origin = origin
        self.sim_id = sim_id
        self.pos = pos
        self.label = label
        self.parent_receptacle = parent_receptacle
        self.parent_aabb = parent_aabb
        self.motion_type = motion_type

class SceneReceptacle:

    def __init__(
            self,   
            name: str, 
            art_type: str = 'rigid',
            origin:Any = None,
            sim_id:int = None, 
            pos:Any = None, 
            label:str = None,
            aabb_receps:List[AABBReceptacle] = None,
            markers:List[Any] = None
    ):
        self.name = name
        self.type = art_type
        self.origin = origin
        self.sim_id = sim_id
        self.pos = pos
        if label:
            self.labels = [label]
        else:
            self.labels = []
        self.aabb_receptacles = aabb_receps
        self.markers = markers

        self.targets = {}
        self.child_objects: List[SceneObject] = []
    
    def add_label(self, label:str):
        self.labels.append(label)
    
    def add_target(self, target, label):
        self.targets[label] = target
    
    def add_child_object(self, obj:SceneObject):
        self.child_objects.append(obj)
    
    def remove_child_object(self, obj:SceneObject):
        self.child_objects.remove(obj)

class SceneInfo:
    objects:List[SceneObject]
    receptacles:List[SceneReceptacle]

    def __init__(
            self,
            objects:List[SceneObject] = [],
            receptacles:List[SceneReceptacle] = [],
    ):
        self.objects = objects
        self.receptacles = receptacles
        self.object_names = []
        self.receptacle_names = []

        if self.objects:
            self.object_names = [obj.name for obj in self.objects]
        if self.receptacles:
            self.receptacle_names = [recep.name for recep in self.receptacles]

    
    def add_object(self, obj:SceneObject):
        self.objects.append(obj)
        self.object_names.append(obj.name)

    def add_receptacle(self, recep:SceneReceptacle):
        self.receptacles.append(recep)
        self.receptacle_names.append(recep.name)
    
    def get_recep_by_name(self, name:str):
        assert name in self.receptacle_names, "Receptacle {} not found".format(name)
        return next((recep for recep in self.receptacles if recep.name == name), None)

    def get_object_by_name(self, name:str):
        assert name in self.object_names, "Object {} not found".format(name)
        return next((obj for obj in self.objects if obj.name == name), None)

@registry.register_sensor
class MarkerVisualizeSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Sensor for visualizing markers on the articulated object
    """
    cls_uuid: str = "marker_visualize"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)

        self._sim = sim
        self.height = config.height
        self.width = config.width
        self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        self._config = config
        self._markers = config.markers
        self._receptacles = config.receptacles
    
    def _get_uuid(self, *args, **kwargs) -> str:
        return MarkerVisualizeSensor.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs) -> SensorTypes:
        return SensorTypes.SEMANTIC
    
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64)
    
    # TODO(YCC): visualize all the markers on the articulated object in camera    
    def get_observation(self, *args, **kwargs):
        atm = self._sim.get_articulated_object_manager()
        rom = self._sim.get_rigid_object_manager()

        return None

@registry.register_sensor
class EnvStartTextSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Sensor to get the start text of the environment
    """
    cls_uuid: str = "env_start_text"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim: RearrangeSim = sim
        self.scene_info = SceneInfo()
    
    def _get_uuid(self, *args, **kwargs):
        return EnvStartTextSensor.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs) -> SensorTypes:
        return SensorTypes.TEXT
    
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Text(max_length=10000)
    
    def initialize_objects(self):
        sim = self._sim
        info = sim.ep_info

        rom = sim.get_rigid_object_manager()
        rigid_obj_handles = rom.get_object_handles()

        atm = sim.get_articulated_object_manager()
        art_obj_handles = atm.get_object_handles()

        assert len(info.name_to_receptacle), "lack of name_to_receptacle info in episode"
        for object_handle, parent_receptacle_handle in info.name_to_receptacle.items():
            parent_receptacle_handle, aabb_handle = parent_receptacle_handle.split("|")

            obj = rom.get_object_by_handle(object_handle)
            obj_label = sim._handle_to_goal_name[object_handle]
            if obj.motion_type == habitat_sim.physics.MotionType.DYNAMIC:
                obj_type = 'dynamic'
            elif obj.motion_type == habitat_sim.physics.MotionType.KINEMATIC:
                obj_type = 'kinematic'
            else:
                obj_type = 'undefined'

            if parent_receptacle_handle in rigid_obj_handles:
                parent_recep = rom.get_object_by_handle(parent_receptacle_handle)
                art_type = 'rigid'
            elif parent_receptacle_handle in art_obj_handles:
                parent_recep = atm.get_object_by_handle(parent_receptacle_handle)
                art_type = 'articulated'
            else:
                raise ValueError(f"Unrecognized receptacle handle: {parent_receptacle_handle}")
            
            if parent_receptacle_handle not in self.scene_info.receptacle_names:
                aabbs = find_aabb_by_receptacle_handle(sim, parent_receptacle_handle)
                markers = find_markers_by_receptacle_handle(sim, parent_receptacle_handle, art_type)

                sim_recep = SceneReceptacle(
                    name=parent_receptacle_handle,
                    art_type=art_type,
                    origin=parent_recep,
                    sim_id=parent_recep.object_id,
                    pos=parent_recep.translation,
                    aabb_receps=aabbs,
                    markers=markers
                )
                self.scene_info.add_receptacle(sim_recep)
            else:
                sim_recep = self.scene_info.get_recep_by_name(parent_receptacle_handle)
                aabbs = sim_recep.aabb_receptacles
            
            parent_aabb = None
            for aabb in aabbs:
                if aabb.name == aabb_handle:
                    parent_aabb = aabb
                    break

            sim_obj = SceneObject(
                name=object_handle,
                origin=obj,
                sim_id=sim._handle_to_object_id[object_handle],
                pos=obj.translation,
                label=obj_label,
                parent_receptacle=sim_recep,
                parent_aabb=parent_aabb,
                motion_type=obj_type
            )
            sim_recep.add_child_object(sim_obj)
            self.scene_info.add_object(sim_obj)
    
    def initialize_receptacles(self):
        """
        Generate the description of the goal receptacles in the scene
        """
        sim = self._sim
        info = sim.ep_info

        rom = sim.get_rigid_object_manager()
        rigid_obj_handles = rom.get_object_handles()
        atm = sim.get_articulated_object_manager()
        art_obj_handles = atm.get_object_handles()

        target_trans = sim._get_target_trans()
        assert len(target_trans), "lack of target_trans info in episode"
        targets = {}
        for target_id, trans in target_trans:
            targets[target_id] = trans
        
        for target_id, recep in enumerate(info.goal_receptacles):
            recep_handle = recep[0]
            assert target_id in targets
            target = targets[target_id]
            target_pos = target.translation
            if recep_handle in rigid_obj_handles:
                parent_recep = rom.get_object_by_handle(recep_handle)
                art_type = 'rigid'
            elif recep_handle in art_obj_handles:
                parent_recep = atm.get_object_by_handle(recep_handle)
                art_type = 'articulated'
            else:
                raise ValueError(f"Unrecognized receptacle handle: {recep_handle}")
            recep_label = f"TARGET_any_targets|{target_id}"

            if recep_handle not in self.scene_info.receptacle_names:
                aabbs = find_aabb_by_receptacle_handle(sim, recep_handle)
                markers = find_markers_by_receptacle_handle(sim, recep_handle, art_type)

                sim_recep = SceneReceptacle(
                    name=recep_handle,
                    art_type=art_type,
                    origin=parent_recep,
                    sim_id=parent_recep.object_id,
                    pos=parent_recep.translation,
                    label=recep_label,
                    aabb_receps=aabbs,
                    markers=markers
                )
                sim_recep.add_target(target_pos, recep_label)
                self.scene_info.add_receptacle(sim_recep)  
            else:
                sim_recep = self.scene_info.get_recep_by_name(recep_handle)      
                sim_recep.add_label(recep_label)
                sim_recep.add_target(target_pos, recep_label)                        

    def get_observation(self, *args, **kwargs):

        self.initialize_objects()
        self.initialize_receptacles()
        objects_description = (
            f"There are {len(self.scene_info.objects)} objects in the scene, "
            "which can be interacted with in various ways.\n"
        )
        objects_description += (
            "For objects with kinematic motion type, {feature}".format(
                feature=motion_type_map["kinematic"]
            )
        )
        objects_description += (
            "For objects with dynamic motion type, {feature}".format(
                feature=motion_type_map["dynamic"]
            )
        )

        receptacles_description = (
            f"There are {len(self.scene_info.receptacles)} receptacles in the scene, "
            "all the receptacles have AABBReceptacles(bounding box) where objects can be placed.\n"
        )
        receptacles_description += (
            "For rigid receptacles, {feature}".format(
                feature=articulated_map["rigid"]
            )
        )
        receptacles_description += (
            "For articulated receptacles, {feature}".format(
                feature=articulated_map["articulated"]
            )
        )

        # add description for each object and receptacle
        for obj in self.scene_info.objects:
            objects_description += generate_object_description(obj)
            
        for recep in self.scene_info.receptacles:
            receptacles_description += generate_receptacle_description(recep)

        scene_description = {
            "objects": objects_description,
            "receptacles": receptacles_description
        }

        scene_description_str = json.dumps(scene_description)

        return scene_description_str

def generate_object_description(obj: SceneObject):
    name = obj.name
    label = obj.label
    parent_aabb_name = obj.parent_aabb.name
    parent_recep = obj.parent_receptacle.name
    art_type = obj.parent_receptacle.type
    obj_type = obj.motion_type

    objects_description = OBJECT_DESCRIPTION.format(
        name=name,
        label=label,
        type=obj_type,
        parent_aabb_name=parent_aabb_name,
        parent_recep=parent_recep,
        space_relation=space_map[art_type],
    )

    return objects_description

def generate_receptacle_description(recep: SceneReceptacle):
    art_type = recep.type
    name = recep.name
    labels = recep.labels
    child_objects = [obj.name for obj in recep.child_objects]
    aabbs = [aabb.name for aabb in recep.aabb_receptacles]
    markers = [marker['name'] for marker in recep.markers]

    if art_type == 'articulated':
        receptacle_description = ARTICULATED_RECEPTACLE_DESCRIPTION.format(
            name=name,
            labels=labels,
            child_objects=str(child_objects),
            aabbs=str(aabbs),
            markers=markers
        )
    else:
        receptacle_description = RIGID_RECEPTACLE_DESCRIPTION.format(
            name=name,
            labels=labels,
            child_objects=str(child_objects),
            aabbs=str(aabbs)
        )

    return receptacle_description


def get_env_start_text(sim, robot_config = None, *args, **kwargs):

    start_sensors = {}
    lab_sensors_config = kwargs.get("lab_sensors_config", None)

    if not lab_sensors_config:
        lab_sensors_config = {}

        lab_sensors_config["robot_resume"] = RobotResumeSensorConfig()
        lab_sensors_config["env_start_text"] = EnvStartTextSensorConfig()

        start_sensors = {
            "robot_resume": RobotResumeSensor(sim, lab_sensors_config["robot_resume"], *args, **kwargs),
            "env_start_text": EnvStartTextSensor(sim, lab_sensors_config["env_start_text"], *args, **kwargs),
        }
    
    start_text = {
        "robot_resume": start_sensors["robot_resume"].get_observation(robot_config, *args, **kwargs),
        "env_start_text": start_sensors["env_start_text"].get_observation(*args, **kwargs),
    }

    return start_text