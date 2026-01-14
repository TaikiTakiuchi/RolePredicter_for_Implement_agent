"""
Pipelines module for Werewolf Role Prediction

This module contains training and prediction pipelines.
"""

from .training_pipeline import main, RolePredictor

__all__ = ['main', 'RolePredictor']
