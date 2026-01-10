"""
2D Tennis Simulator

A simulation environment for AI agents to play tennis.
Designed to observe emergent tactical behaviors.
"""

from ball import Ball, create_serve_ball
from config import DEFAULT_CONFIG, Config
from field import Field, Rectangle
from game import Game, GameState, PointResult, StepResult
from player import NUM_MOVEMENT_ACTIONS, Player, create_players

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "Field",
    "Rectangle",
    "Ball",
    "create_serve_ball",
    "Player",
    "create_players",
    "NUM_MOVEMENT_ACTIONS",
    "Game",
    "GameState",
    "PointResult",
    "StepResult",
]
