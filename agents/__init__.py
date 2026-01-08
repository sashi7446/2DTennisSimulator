"""
Agent system for 2D Tennis Simulator.

Provides a unified interface for different AI agents.
"""

from agents.base import Agent, AgentConfig, load_agent
from agents.chase import ChaseAgent, SmartChaseAgent
from agents.random_agent import RandomAgent
from agents.positional import PositionalAgent

# NeuralAgent requires numpy
try:
    from agents.neural import NeuralAgent
    NEURAL_AVAILABLE = True
except ImportError:
    NeuralAgent = None
    NEURAL_AVAILABLE = False

__all__ = [
    "Agent",
    "AgentConfig",
    "load_agent",
    "ChaseAgent",
    "SmartChaseAgent",
    "RandomAgent",
    "PositionalAgent",
    "NeuralAgent",
    "NEURAL_AVAILABLE",
]
