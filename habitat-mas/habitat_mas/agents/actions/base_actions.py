import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from habitat.tasks.rearrange.actions.oracle_nav_action import (
    OracleNavCoordinateAction,
    OracleNavAction
)

from ..crab_core import action

# TODO: to connect with the habitat-lab/ habitat-mas agents. 

@action
def get_agents() -> List[str]:
    """
    Get the list of agents in the environment.
    """
    pass

@action
def send_request(request: str, target_agent: str) -> str:
    """
    Send a text request to the fellow agents.
    
    Args:
        request: The text request to send.
        target_agent: The agent to send the request to.
    
    """
    pass

@action
def wait():
    """
    Wait if you don't have to take any action.
    """
    pass

@action
def open_fridge():
    """
    Open the fridge. Only when you are near the fridge.
    """
    pass

@action
def close_fridge():
    """
    Close the fridge. Only when you are near the fridge.
    """
    pass

@action
def open_cab():
    """
    Open the cabinet. Only when you are near the cabinet.
    """
    pass

@action
def close_cab():
    """
    Close the cabinet. Only when you are near the cabinet.
    """
    pass

@action
def nav_to_obj(target_obj: str):
    """
    Navigate to an object.
    
    Args:
        target_obj: The object to navigate to.
    """
    pass

@action
def nav_to_goal(goal: str):
    """
    Navigate to a goal.
    
    Args:
        goal: The goal to navigate to.
    """
    pass
