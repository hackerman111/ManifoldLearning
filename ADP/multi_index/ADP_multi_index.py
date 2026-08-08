
from dataclasses import dataclass

from ADP import ADP_Config, ADP_solver


@dataclass(slots=True)
class ADP_single_index:
    config: ADP_Config
    solver: ADP_solver
    
