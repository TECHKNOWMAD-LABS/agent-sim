from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseEnvironment

EMPTY: int = 0
WALL: int = 1
FOOD: int = 2

DELTAS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
    "collect": (0, 0),
}

_SYMBOLS = {EMPTY: ".", WALL: "#", FOOD: "F"}


@dataclass
class _AgentRecord:
    position: tuple[int, int]
    collected: int = 0
    alive: bool = True


class GridEnvironment(BaseEnvironment):
    """2D grid world supporting walls, food resources, and multiple agents.

    Cell encoding
    -------------
    0 = empty, 1 = wall, 2 = food
    Agents are tracked separately (not embedded in the grid array).
    """

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        food_count: int = 5,
        wall_density: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"rows must be a positive integer, got {rows!r}")
        if not isinstance(cols, int) or cols < 1:
            raise ValueError(f"cols must be a positive integer, got {cols!r}")
        if not isinstance(food_count, int) or food_count < 0:
            raise ValueError(f"food_count must be a non-negative integer, got {food_count!r}")
        if not isinstance(wall_density, (int, float)) or not (0.0 <= wall_density < 1.0):
            raise ValueError(
                f"wall_density must be in [0.0, 1.0), got {wall_density!r}"
            )
        self.rows = rows
        self.cols = cols
        self.food_count = food_count
        self.wall_density = wall_density
        self._rng = np.random.default_rng(seed)
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=np.int8)
        self._agents: dict[str, _AgentRecord] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: str, position: tuple[int, int] | None = None) -> None:
        """Register an agent.  Call before *reset*.

        Args:
            agent_id: Unique string identifier for the agent.
            position: Optional (row, col) starting position.  If None a random
                empty cell is chosen.

        Raises:
            ValueError: If agent_id is not a non-empty string or if position
                is out of bounds.
        """
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError(f"agent_id must be a non-empty string, got {agent_id!r}")
        if position is not None:
            r, c = position
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(
                    f"position {position!r} is out of bounds for {self.rows}x{self.cols} grid"
                )
            self._agents[agent_id] = _AgentRecord(position=position)
        else:
            self._agents[agent_id] = _AgentRecord(position=self._random_empty())

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset the grid and reposition all registered agents.

        Returns:
            Mapping of agent_id to initial observation dict.
        """
        self._grid[:] = EMPTY
        self._step_count = 0
        self._place_walls()
        self._place_food()
        for record in self._agents.values():
            record.position = self._random_empty()
            record.collected = 0
            record.alive = True
        return {aid: self.get_observation(aid) for aid in self._agents}

    def step(self, agent_id: str, action: str) -> tuple[dict[str, Any], float, bool]:
        """Execute *action* for *agent_id*.

        Args:
            agent_id: ID of the acting agent.
            action: One of the recognised action strings (unknown actions are
                treated as 'stay').

        Returns:
            Tuple of (observation, reward, done).

        Raises:
            KeyError: If agent_id is not registered.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent '{agent_id}'. Register via add_agent() first.")
        if not isinstance(action, str):
            action = "stay"

        record = self._agents[agent_id]
        reward = -0.01  # small step penalty encourages efficiency
        done = False

        r, c = record.position
        dr, dc = DELTAS.get(action, (0, 0))

        if action == "collect":
            if self._grid[r, c] == FOOD:
                self._grid[r, c] = EMPTY
                record.collected += 1
                reward = 1.0
        else:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self._grid[nr, nc] != WALL:
                record.position = (nr, nc)

        self._step_count += 1
        if not np.any(self._grid == FOOD):
            done = True

        return self.get_observation(agent_id), reward, done

    def get_observation(self, agent_id: str) -> dict[str, Any]:
        """Return the current observation for *agent_id* without advancing time.

        Args:
            agent_id: ID of the agent.

        Raises:
            KeyError: If agent_id is not registered.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent '{agent_id}'. Register via add_agent() first.")
        record = self._agents[agent_id]
        r, c = record.position
        local_view = self._local_view(r, c)
        food_dir = self._nearest_food_direction(r, c)
        return {
            "position": record.position,
            "local_view": local_view,
            "food_nearby": bool(self._grid[r, c] == FOOD),
            "food_direction": food_dir,
            "collected": record.collected,
            "grid_size": self.cols,
            "step": self._step_count,
        }

    def render(self) -> str:
        """Return a human-readable ASCII representation of the environment."""
        agent_cells = {rec.position: aid[0].upper() for aid, rec in self._agents.items()}
        lines: list[str] = []
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r, c) in agent_cells:
                    row += agent_cells[(r, c)]
                else:
                    row += _SYMBOLS.get(int(self._grid[r, c]), "?")
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def food_remaining(self) -> int:
        return int(np.sum(self._grid == FOOD))

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_walls(self) -> None:
        n_walls = int(self.rows * self.cols * self.wall_density)
        for _ in range(n_walls):
            r, c = self._random_empty()
            self._grid[r, c] = WALL

    def _place_food(self) -> None:
        for _ in range(self.food_count):
            r, c = self._random_empty()
            self._grid[r, c] = FOOD

    def _random_empty(self) -> tuple[int, int]:
        for _ in range(1000):
            r = int(self._rng.integers(0, self.rows))
            c = int(self._rng.integers(0, self.cols))
            if self._grid[r, c] == EMPTY:
                return (r, c)
        # Exhaustive fallback: scan every cell
        for r in range(self.rows):
            for c in range(self.cols):
                if self._grid[r, c] == EMPTY:
                    return (r, c)
        return (0, 0)

    def _local_view(self, r: int, c: int) -> list[list[int]]:
        local: list[list[int]] = []
        for dr in range(-1, 2):
            row: list[int] = []
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    row.append(int(self._grid[nr, nc]))
                else:
                    row.append(WALL)
            local.append(row)
        return local

    def _nearest_food_direction(self, r: int, c: int) -> str | None:
        food_coords = list(zip(*np.where(self._grid == FOOD)))
        if not food_coords:
            return None
        nearest = min(food_coords, key=lambda p: abs(int(p[0]) - r) + abs(int(p[1]) - c))
        fr, fc = int(nearest[0]), int(nearest[1])
        if abs(fr - r) > abs(fc - c):
            return "up" if fr < r else "down"
        if fc != c:
            return "left" if fc < c else "right"
        return None
