"""Cost tracking for API calls (OpenRouter, HuggingFace API, etc.)."""

from __future__ import annotations


class CostTracker:
    """Accumulates token usage and costs across pipeline steps."""

    def __init__(self):
        self.entries: list[dict] = []

    def add(
        self,
        step: str,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float | None = None,
    ):
        self.entries.append({
            "step": step,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
            "cost_usd": cost_usd,
        })

    @property
    def total_tokens(self) -> int:
        return sum(e["total_tokens"] for e in self.entries)

    @property
    def total_cost(self) -> float | None:
        costs = [e["cost_usd"] for e in self.entries if e["cost_usd"] is not None]
        return sum(costs) if costs else None

    def summary(self):
        """Return a summary DataFrame grouped by step."""
        import pandas as pd

        if not self.entries:
            return pd.DataFrame(columns=["step", "provider", "model", "total_tokens", "cost_usd"])

        df = pd.DataFrame(self.entries)
        return (
            df.groupby(["step", "provider", "model"], sort=False)
            .agg(
                prompt_tokens=("prompt_tokens", "sum"),
                completion_tokens=("completion_tokens", "sum"),
                total_tokens=("total_tokens", "sum"),
                cost_usd=("cost_usd", lambda s: s.sum(min_count=1)),
                calls=("step", "count"),
            )
            .reset_index()
        )

    def __repr__(self) -> str:
        if not self.entries:
            return "CostTracker: no entries"
        import pandas as pd

        lines = ["CostTracker Summary", "=" * 60]
        for _, row in self.summary().iterrows():
            cost_str = f"${row['cost_usd']:.4f}" if pd.notna(row["cost_usd"]) else "n/a"
            lines.append(
                f"  {row['step']:20s} | {row['model']:30s} | "
                f"{row['total_tokens']:>8,} tokens | {cost_str}"
            )
        total_cost = self.total_cost
        cost_line = f"${total_cost:.4f}" if total_cost is not None else "n/a"
        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':20s} | {'':30s} | {self.total_tokens:>8,} tokens | {cost_line}")
        return "\n".join(lines)
