"""
2D Tennis Simulator

A simulation environment for AI agents to play tennis.
Designed to observe emergent tactical behaviors.
"""

from config import Config, DEFAULT_CONFIG
from field import Field, Rectangle
from ball import Ball, create_serve_ball
from player import Player, create_players, NUM_MOVEMENT_ACTIONS
from game import Game, GameState, PointResult, StepResult

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
