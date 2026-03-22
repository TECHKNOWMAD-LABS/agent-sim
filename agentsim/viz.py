from __future__ import annotations

from .environment.grid import GridEnvironment
from .simulation import EpisodeResult


def render_grid_ascii(env: GridEnvironment) -> str:
    """Return the current grid state as an ASCII string."""
    return env.render()


def render_episode_summary(result: EpisodeResult) -> str:
    """One-paragraph text summary of a single episode result."""
    lines = [
        f"Episode {result.episode}",
        f"  Steps : {result.steps}",
        f"  Done  : {result.done}",
        "  Rewards:",
    ]
    for aid, reward in result.total_reward.items():
        lines.append(f"    {aid}: {reward:+.3f}")
    return "\n".join(lines)


def plot_reward_trend(reward_trend: list[float], title: str = "Reward Trend") -> str:
    """Render a Unicode sparkline for the reward series."""
    if not reward_trend:
        return f"{title}: (no data)"
    mn = min(reward_trend)
    mx = max(reward_trend)
    span = mx - mn or 1.0
    bars = " ▁▂▃▄▅▆▇█"
    sparkline = "".join(
        bars[min(int(((v - mn) / span) * (len(bars) - 1)), len(bars) - 1)]
        for v in reward_trend
    )
    return f"{title}: {sparkline}  [{mn:.2f} … {mx:.2f}]"


def render_agent_heatmap(
    positions: list[tuple[int, int]],
    rows: int,
    cols: int,
) -> str:
    """ASCII density heatmap of agent visitation frequency."""
    grid = [[0] * cols for _ in range(rows)]
    for r, c in positions:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] += 1
    symbols = " ░▒▓█"
    max_val = max(max(row) for row in grid) or 1
    lines = [
        "".join(
            symbols[min(int((val / max_val) * (len(symbols) - 1)), len(symbols) - 1)]
            for val in row
        )
        for row in grid
    ]
    return "\n".join(lines)


def simulation_report(
    results: list[EpisodeResult],
    reward_trend: list[float] | None = None,
) -> str:
    """Compile a full plain-text simulation report."""
    sep = "=" * 42
    lines = [
        sep,
        "  AgentSim Simulation Report",
        sep,
        f"  Episodes  : {len(results)}",
        f"  Completed : {sum(1 for r in results if r.done)}",
    ]
    if reward_trend:
        lines.append("  " + plot_reward_trend(reward_trend))
    lines.append("-" * 42)
    for result in results[:5]:
        for line in render_episode_summary(result).splitlines():
            lines.append("  " + line)
    if len(results) > 5:
        lines.append(f"  … and {len(results) - 5} more episodes")
    lines.append(sep)
    return "\n".join(lines)
