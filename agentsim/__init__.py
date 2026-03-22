"""AgentSim — multi-agent simulation environment."""

from .agents.base import AgentState, BaseAgent
from .agents.deliberative import DeliberativeAgent, Goal
from .agents.learning import LearningAgent
from .agents.reactive import ReactiveAgent
from .analysis import compute_metrics, compute_trajectory_stats
from .environment.grid import GridEnvironment
from .scenarios.foraging import ForagingResult, ForagingScenario, make_forager
from .scenarios.pursuit import PursuitResult, PursuitScenario, make_predator, make_prey
from .simulation import EpisodeResult, Simulation, SimulationConfig
from .viz import render_grid_ascii, simulation_report

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "BaseAgent",
    "DeliberativeAgent",
    "EpisodeResult",
    "ForagingResult",
    "ForagingScenario",
    "Goal",
    "GridEnvironment",
    "LearningAgent",
    "PursuitResult",
    "PursuitScenario",
    "ReactiveAgent",
    "Simulation",
    "SimulationConfig",
    "compute_metrics",
    "compute_trajectory_stats",
    "make_forager",
    "make_predator",
    "make_prey",
    "render_grid_ascii",
    "simulation_report",
]
