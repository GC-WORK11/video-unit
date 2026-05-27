"""Registry for VideoUnit evaluators."""

from typing import Callable

from ._base import Evaluator

EVALUATORS: dict[str, type[Evaluator]] = {}


def register_evaluator(cls: type[Evaluator]) -> type[Evaluator]:
    """Decorator to register an evaluator class.

    The evaluator's `name` attribute is used as the registration key.
    Registered evaluators can be retrieved by name using `get_evaluator()`.

    Args:
        cls: The evaluator class to register.

    Returns:
        The same class after registration.

    Raises:
        TypeError: If cls is not a subclass of Evaluator.
        ValueError: If cls.name is already registered.

    Example:
        @register_evaluator
        class MyEvaluator(Evaluator):
            name = "my_evaluator"
            ...
    """
    if not issubclass(cls, Evaluator):
        raise TypeError(f"{cls.__name__} must be a subclass of Evaluator")

    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name' attribute")

    if cls.name in EVALUATORS:
        raise ValueError(
            f"Evaluator name '{cls.name}' is already registered to {EVALUATORS[cls.name].__name__}"
        )

    EVALUATORS[cls.name] = cls
    return cls


def get_evaluator(name: str) -> Evaluator:
    """Get an instance of the evaluator with the given name.

    Args:
        name: The evaluator's unique name identifier.

    Returns:
        A new instance of the requested evaluator.

    Raises:
        ValueError: If no evaluator with that name is registered.

    Example:
        evaluator = get_evaluator("object_exists")
        result = await evaluator.run(context)
    """
    if name not in EVALUATORS:
        available = list(EVALUATORS.keys())
        raise ValueError(
            f"Unknown evaluator: '{name}'. Available evaluators: {available}"
        )
    return EVALUATORS[name]()


def all_evaluators() -> list[str]:
    """Get the names of all registered evaluators.

    Returns:
        Alphabetically sorted list of registered evaluator names.
    """
    return sorted(EVALUATORS.keys())


def evaluator_count() -> int:
    """Get the number of registered evaluators.

    Returns:
        Number of evaluators in the registry.
    """
    return len(EVALUATORS)
