"""
Agent system for 2D Tennis Simulator.

Provides a unified interface for different AI agents.
"""

from agents.base import Agent, AgentConfig, load_agent
from agents.baseliner import BaselinerAgent
from agents.chase import ChaseAgent, SmartChaseAgent
from agents.positional import PositionalAgent
from agents.random_agent import RandomAgent

# NeuralAgent requires numpy
try:
    from agents.neural import NeuralAgent
    from agents.transformer import TransformerAgent

    NEURAL_AVAILABLE = True
except ImportError:
    NeuralAgent = None
    TransformerAgent = None
    NEURAL_AVAILABLE = False

__all__ = [
    "Agent",
    "AgentConfig",
    "load_agent",
    "ChaseAgent",
    "SmartChaseAgent",
    "RandomAgent",
    "PositionalAgent",
    "BaselinerAgent",
    "NeuralAgent",
    "TransformerAgent",
    "NEURAL_AVAILABLE",
]
