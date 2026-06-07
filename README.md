# AI-Venture-Simulation-Environment

This project is an agent-based simulation designed to model the incentive structures, market dynamics, and quality accrual processes within the Liberata academic publishing platform. Agents must strategically allocate their day between advancing their own research and participating in a peer-review marketplace. B

## Short Term Plan - Simulate bad faith peer review
The simulation presents the choice of good faith and bad faith peer reviews, and seeks to observe agentic behavior of making these choices.

## Review Effort Model

Peer review quality is modeled through continuous effort. A review must pass `MIN_REVIEW_EFFORT_THRESHOLD` to count as completed. Completed reviews below `GOOD_FAITH_REVIEW_EFFORT_THRESHOLD` are classified as bad faith; completed reviews at or above that threshold are classified as good faith.

Reviewer share does not depend on good-faith/bad-faith classification. The classification only affects the paper's future accrual rate.

## Writing Effort Model

Paper writing effort is continuous. Each `write_paper` action contributes a `writing_effort_delta` to the agent's current paper progress. Once cumulative progress reaches `PAPER_THRESHOLD`, the agent publishes a paper and progress resets.

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

## Logging runs
Separately from the live dashboard, you can save completed runs and browse them later in a static web page . This is gets published to GitHub
Pages.

After running the simulation, the terminal will prompt you whether or not to save this run to the log and ask for a name.
