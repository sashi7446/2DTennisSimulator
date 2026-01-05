"""
Agent system for 2D Tennis Simulator.

Provides a unified interface for different AI agents.
"""

from agents.base import Agent, AgentConfig
from agents.chase import ChaseAgent
from agents.random_agent import RandomAgent

__all__ = [
    "Agent",
    "AgentConfig",
    "ChaseAgent",
    "RandomAgent",
]
