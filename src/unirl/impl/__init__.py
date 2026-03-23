"""UniRL implementation package.

Concrete, torch-based implementations of the UniRL core protocols live here.

Sub-packages
------------
agents
    Concrete :class:`~unirl.core.agent.Agent` implementations.
rollouts
    Concrete :class:`~unirl.core.rollout.Rollout` implementations.
batch_sources
    Concrete :class:`~unirl.core.batch_source.BatchSource` implementations.
learners
    Concrete :class:`~unirl.core.learner.Learner` implementations.
models
    Shared neural network components (torch ``nn.Module`` subclasses).
"""
