# AI-Venture-Simulation-Environment

This project is an agent-based simulation designed to model the incentive structures, market dynamics, and quality accrual processes within the Liberata academic publishing platform. Agents must strategically allocate their day between advancing their own research and participating in a peer-review marketplace. B

## Short Term Plan - Simulate bad faith peer review
The simulation presents the choice of good faith and bad faith peer reviews, and seeks to observe agentic behavior of making these choices.

## Features
Our environment stresses a few main features:
- Flexible interfaces for agent, environment, market, and paper classes. This allows for multiple implementations of various algorithms.
- Various methods for more complex simulation. This can be chosen to be turned on or off depending on the simulation we want to run.

## Live dashboard
Watch a run unfold in the browser.

```
python dashboard_server.py
```

Then open http://127.0.0.1:8000. The server is stdlib-only and runs entirely on
localhost. Each browser connection streams a fresh, deterministic run with a small delay every step so the run is watchable.