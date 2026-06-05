"""Streaming variant of :class:`History` for the live dashboard.

``StreamingHistory`` is a drop-in ``History`` (so ``Environment`` uses it
unchanged) that, on top of recording the usual series, pushes a compact per-day
payload onto a thread-safe queue each time the environment advances a day. The
dashboard server drains that queue and forwards each payload to the browser over
Server-Sent Events. No changes to the core simulation are required.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

from History import History

if TYPE_CHECKING:
    from Environment import Environment


class StreamingHistory(History):
    """``History`` that also emits a per-day snapshot onto ``self.queue``."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._last_action_idx = 0

    def record_step(self, env: "Environment") -> None:
        # Populate all the normal series first, then read the freshly appended
        # tail values to build a compact delta payload for this day.
        super().record_step(env)

        payload: dict[str, Any] = {
            "type": "step",
            "day": self.days[-1],
            "agents": {
                label: series[-1] for label, series in self.agent_capital.items()
            },
            "scalars": {name: vals[-1] for name, vals in self.scalars.items()},
            "actions": [
                {"agent": agent, "kind": kind, "paper": paper}
                for (_, agent, kind, paper) in self.actions[self._last_action_idx:]
            ],
        }
        self._last_action_idx = len(self.actions)
        self.queue.put(payload)
