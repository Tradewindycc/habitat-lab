import os
from typing import List, Dict, Tuple, Union
import numpy as np
from habitat import Env

from ..crab_core import action


@action
def pick(target_obj: str):
    """
    Pick an object.
    
    Args:
        target_obj: The object to pick.
    """
    pass


@action
def place(target_obj: str, target_location: str):
    """
    Place an object.
    
    Args:
        target_obj: The object to place.
        target_location: The location to place the object.
    """
    pass


@action
def reset_arm():
    """
    Reset Arm.
    """
    pass
