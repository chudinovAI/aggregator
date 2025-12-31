"""
FSM states for bot conversation flows.
"""

from aiogram.fsm.state import State, StatesGroup


class TopicStates(StatesGroup):
    """States for topic management flow."""

    waiting_for_topic = State()


class SourceStates(StatesGroup):
    """States for source management flow."""

    selecting_sources = State()


class PeriodStates(StatesGroup):
    """States for period selection flow."""

    selecting_period = State()


__all__ = [
    "TopicStates",
    "SourceStates",
    "PeriodStates",
]
