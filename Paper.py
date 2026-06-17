from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agent import Agent

from config import SIM

# Bound from SIM (config.py is the single source of truth); names kept for the
# many internal references and default-argument signatures below.
DEFAULT_ACCRUAL_RATE = SIM.default_accrual_rate
DEFAULT_REVIEW_SHARE = SIM.default_review_share
USE_FAIR_MARKET_PRICING = SIM.use_fair_market_pricing
PRIOR_REVIEW_EPSILON = SIM.prior_review_epsilon
MIN_REVIEW_EFFORT_THRESHOLD = SIM.min_review_effort_threshold
GOOD_FAITH_REVIEW_THRESHOLD = SIM.good_faith_review_threshold
REVIEW_EFFORT_PER_TIMESTEP = SIM.review_effort_per_timestep
BAD_REVIEW_TIMESTEPS = SIM.bad_review_timesteps
GOOD_REVIEW_TIMESTEPS = SIM.good_review_timesteps
DISCRETE_PAPER_TIMESTEPS = SIM.discrete_paper_timesteps
DISCRETE_WRITING_EFFORT_PER_TIMESTEP = SIM.discrete_writing_effort_per_timestep
REVIEW_EFFORT_CURVE = SIM.review_effort_curve
MIN_REVIEW_ACCRUAL_BUMP = SIM.min_review_accrual_bump
MAX_REVIEW_ACCRUAL_BUMP = SIM.max_review_accrual_bump
REVIEW_SIGMOID_MIDPOINT = SIM.review_sigmoid_midpoint
REVIEW_SIGMOID_STEEPNESS = SIM.review_sigmoid_steepness
REVIEW_JUMP_THRESHOLD = SIM.review_jump_threshold
REVIEW_JUMP_BUMP = SIM.review_jump_bump
REVIEW_JUMP_WIDTH = SIM.review_jump_width
BASE_REVIEW_ACCRUAL_BUMP = SIM.base_review_accrual_bump
FIRST_EXTRA_DAY_BUMP = SIM.first_extra_day_bump
DEFAULT_MAX_REVIEWER_SHARE = SIM.default_max_reviewer_share
MIN_OFFER_SHARE = SIM.min_offer_share
QUALITY_SIGMA = SIM.quality_sigma
MIN_PAPER_QUALITY = SIM.min_paper_quality
QUALITY_PRICE_SCALE = SIM.quality_price_scale
HISTORY_PRICE_SCALE = SIM.history_price_scale
WRITING_SATURATION = SIM.writing_saturation

REVIEW_PARADIGM_CONTINUOUS = "continuous"
REVIEW_PARADIGM_DISCRETE = "discrete"
VALID_REVIEW_PARADIGMS = frozenset({
    REVIEW_PARADIGM_CONTINUOUS,
    REVIEW_PARADIGM_DISCRETE,
})

BAD_FAITH_REVIEW = "bad_faith"
GOOD_FAITH_REVIEW = "good_faith"
VALID_REVIEW_KINDS = frozenset({BAD_FAITH_REVIEW, GOOD_FAITH_REVIEW})


def validate_review_paradigm(paradigm: str) -> str:
    value = str(paradigm).strip().lower()
    if value not in VALID_REVIEW_PARADIGMS:
        allowed = ", ".join(sorted(VALID_REVIEW_PARADIGMS))
        raise ValueError(f"review_paradigm must be one of: {allowed}")
    return value


def normalize_review_kind(review_kind: str | None) -> str:
    value = str(review_kind or "").strip().lower()
    if value not in VALID_REVIEW_KINDS:
        allowed = ", ".join(sorted(VALID_REVIEW_KINDS))
        raise ValueError(f"review_kind must be one of: {allowed}")
    return value


def review_kind_from_effort(effort: float) -> str:
    """Classify a completed review by the continuous-mode threshold."""
    return (
        GOOD_FAITH_REVIEW
        if float(effort) >= GOOD_FAITH_REVIEW_THRESHOLD
        else BAD_FAITH_REVIEW
    )


def fixed_review_effort(review_kind: str) -> float:
    """Discrete-mode duration for a chosen review kind."""
    kind = normalize_review_kind(review_kind)
    if kind == GOOD_FAITH_REVIEW:
        return GOOD_REVIEW_TIMESTEPS
    return BAD_REVIEW_TIMESTEPS


def review_action_kind(review_kind: str) -> str:
    """Action-log label for a completed good/bad-faith review."""
    return f"{normalize_review_kind(review_kind)}_review"


def quality_multiplier(quality: float) -> float:
    """Clamp paper quality to a strictly positive multiplier."""
    return max(MIN_PAPER_QUALITY, float(quality))


def accrual_rate_from_quality(quality: float) -> float:
    """Base AC accrual rate implied by a paper's quality (the asymptotic ceiling)."""
    return DEFAULT_ACCRUAL_RATE * quality_multiplier(quality)


def accrual_rate_from_effort(quality: float, writing_effort: float) -> float:
    """Continuous-mode base accrual rate after ``writing_effort`` timesteps.

    The rate approaches the quality-defined ceiling
    ``accrual_rate_from_quality(quality)`` asymptotically:
    ``ceiling * (1 - exp(-WRITING_SATURATION * effort))``. Zero effort yields a
    zero rate; more effort moves the paper closer to its ceiling with
    diminishing returns. There is no minimum threshold to finish a paper.
    """
    effort = max(0.0, float(writing_effort))
    ceiling = accrual_rate_from_quality(quality)
    return ceiling * (1.0 - math.exp(-WRITING_SATURATION * effort))


def fair_market_component(epsilon: float) -> float:
    """One reviewer's fair-market share term: epsilon / (1 + epsilon)."""
    value = max(0.0, float(epsilon))
    return value / (1.0 + value)


def fair_market_price_from_epsilons(
    epsilons,
    prior_epsilon: float = PRIOR_REVIEW_EPSILON,
) -> float:
    """Expected fair-market price from the empirical reviewer epsilon distribution.

    This implements the schematic's formula
    ``sum(epsilon / (1 + epsilon) * Probability(epsilon))`` by treating the
    current reviewer epsilon list as an equally weighted empirical distribution.
    """
    cleaned = []
    for epsilon in epsilons:
        try:
            value = float(epsilon)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value) or value < 0.0:
            continue
        cleaned.append(value)

    if not cleaned:
        cleaned = [max(0.0, float(prior_epsilon))]

    return sum(fair_market_component(value) for value in cleaned) / len(cleaned)


def _normalized_sigmoid(effort: float) -> float:
    """Map effort into [0, 1] between one timestep and the long-review target."""
    lower = MIN_REVIEW_EFFORT_THRESHOLD
    upper = max(lower + 1e-9, GOOD_REVIEW_TIMESTEPS)
    steepness = max(1e-9, REVIEW_SIGMOID_STEEPNESS)
    midpoint = REVIEW_SIGMOID_MIDPOINT

    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

    lo = sig(lower)
    hi = sig(upper)
    if abs(hi - lo) < 1e-12:
        return 1.0 if effort >= upper else 0.0
    scaled = (sig(effort) - lo) / (hi - lo)
    return min(1.0, max(0.0, scaled))


def review_accrual_bump(effort: float, quality: float = 1.0) -> float:
    """Accrual-rate bump fraction for a single review of ``effort`` and ``quality``.

    Effort below ``MIN_REVIEW_EFFORT_THRESHOLD`` yields 0. The default sigmoid
    mode approximates the review-length evidence discussed by the team: short
    reviews have limited effect, impact rises through the good-faith region, and
    long reviews saturate. The previous logarithmic curve remains available by
    setting ``SIM.review_effort_curve = "log"``; ``"jump"`` is an optional
    experiment for a high-effort threshold bonus discussed in sync.
    """
    if effort < MIN_REVIEW_EFFORT_THRESHOLD:
        return 0.0

    curve = str(REVIEW_EFFORT_CURVE).strip().lower()
    if curve == "log":
        base = BASE_REVIEW_ACCRUAL_BUMP * quality_multiplier(quality)
        extra = effort - MIN_REVIEW_EFFORT_THRESHOLD
        return base + FIRST_EXTRA_DAY_BUMP * math.log2(1 + extra)

    span = max(0.0, MAX_REVIEW_ACCRUAL_BUMP - MIN_REVIEW_ACCRUAL_BUMP)
    bump = MIN_REVIEW_ACCRUAL_BUMP + span * _normalized_sigmoid(float(effort))
    if curve == "jump":
        width = max(1e-9, float(REVIEW_JUMP_WIDTH))
        jump = 1.0 / (1.0 + math.exp(-(float(effort) - REVIEW_JUMP_THRESHOLD) / width))
        bump += max(0.0, REVIEW_JUMP_BUMP) * jump
    return bump * quality_multiplier(quality)


def review_epsilon_from_effort(effort: float, quality: float = 1.0) -> float:
    """Public helper for the review-induced proportional accrual improvement."""
    return review_accrual_bump(effort, quality)


class Paper:
    """A paper in the single-review marketplace.

    A paper is created with a known ``quality`` and is published to the market
    one timestep after its author finishes writing it. While listed, its author
    offers each potential reviewer a distinct share price (``price_table``). The
    first agent to claim it takes it permanently off the market; the paper can be
    reviewed exactly once.
    """

    def __init__(
        self,
        author: Agent,
        quality: float = 1.0,
        accrual_rate: float | None = None,
        current_ac: float = 0.0,
        share_distribution: dict[Agent, float] | None = None,
        completion_progress: float = 1.0,
        market_listed: bool = False,
        max_reviewer_share: float = DEFAULT_MAX_REVIEWER_SHARE,
        writing_effort: float | None = None,
        required_writing_effort: float | None = None,
    ):
        if author is None:
            raise ValueError("author cannot be None")

        self.author = author
        self.quality = quality_multiplier(quality)
        self.writing_effort = (
            None if writing_effort is None else max(0.0, float(writing_effort))
        )
        self.required_writing_effort = (
            None
            if required_writing_effort is None
            else max(0.0, float(required_writing_effort))
        )
        if accrual_rate is not None:
            rate = accrual_rate
        elif self.writing_effort is not None:
            # Continuous mode: writing effort sets how close to the ceiling we get.
            rate = accrual_rate_from_effort(self.quality, self.writing_effort)
        else:
            # Discrete / back-compat: full quality-defined rate.
            rate = accrual_rate_from_quality(self.quality)
        self.accrual_rate = self._nonnegative_float(rate, "accrual_rate")
        self.current_ac = self._nonnegative_float(current_ac, "current_ac")
        self.share_distribution = (
            {author: 1.0} if share_distribution is None else dict(share_distribution)
        )
        self._validate_share_distribution()
        self.completion_progress = self._nonnegative_float(
            completion_progress,
            "completion_progress",
        )
        self.max_reviewer_share = self._validate_share_value(
            max_reviewer_share,
            "max_reviewer_share",
        )

        # Marketplace / single-review lifecycle.
        self.market_listed = bool(market_listed)
        self.scheduled_listing_timestep: int | None = None
        self.review_claimed = False
        self.reviewed = False
        self.reviewer: Agent | None = None
        self.review_in_progress_by: Agent | None = None
        self.agreed_review_share = 0.0
        self.price_table: dict[Agent, float] = {}
        self.review_records: list[dict[str, object]] = []

    # ---- compatibility aliases ------------------------------------------
    @property
    def ac_accrual_rate(self) -> float:
        return self.accrual_rate

    @ac_accrual_rate.setter
    def ac_accrual_rate(self, value: float):
        self.accrual_rate = self._nonnegative_float(value, "ac_accrual_rate")

    @property
    def completed_peer_reviews(self) -> int:
        """0 or 1 — a paper can be reviewed at most once."""
        return 1 if self.reviewed else 0

    @property
    def reviewed_by(self) -> set[Agent]:
        return {self.reviewer} if self.reviewer is not None else set()

    @property
    def review_available(self) -> bool:
        """True when the paper is listed and not yet claimed/reviewed."""
        return self.market_listed and not self.review_claimed and not self.reviewed

    # ---- pricing --------------------------------------------------------
    def update_price_table(
        self,
        reviewers,
        market_median_quality: float,
        mean_peer_review_epsilon: float,
        fair_market_price: float | None = None,
    ) -> None:
        """Recompute the per-reviewer share offer for this listed paper.

        Higher paper quality (relative to the market) lowers the offered share;
        a stronger reviewer epsilon history raises it. The base price is the
        diagram's fair-market expectation ``E[epsilon / (1 + epsilon)]`` unless
        disabled in config. The result is clamped to ``[min_offer_share, author
        share]`` and the single-review cap.
        """
        author_share = self.share_distribution.get(self.author, 0.0)
        ceiling = min(self.max_reviewer_share, max(0.0, author_share))
        if fair_market_price is None:
            fair_market_price = fair_market_price_from_epsilons([PRIOR_REVIEW_EPSILON])
        base_offer = fair_market_price if USE_FAIR_MARKET_PRICING else DEFAULT_REVIEW_SHARE
        table: dict[Agent, float] = {}
        for agent in reviewers:
            if agent is self.author:
                continue
            quality_factor = math.exp(
                -QUALITY_PRICE_SCALE * (self.quality - market_median_quality)
            )
            history = getattr(agent, "peer_review_epsilon_history", PRIOR_REVIEW_EPSILON)
            history_factor = max(
                0.0,
                1.0 + HISTORY_PRICE_SCALE * (history - mean_peer_review_epsilon),
            )
            offer = base_offer * quality_factor * history_factor
            offer = min(max(offer, MIN_OFFER_SHARE), ceiling)
            table[agent] = offer
        self.price_table = table

    def offered_share(self, agent: Agent) -> float:
        """The share currently offered to ``agent`` (0 if none/ineligible)."""
        return float(self.price_table.get(agent, 0.0))

    # ---- single-review lifecycle ----------------------------------------
    def can_start_review(self, agent: Agent) -> bool:
        if agent is self.author:
            return False
        if self.reviewed or self.review_claimed:
            return False
        return self.market_listed

    def start_review(self, agent: Agent) -> bool:
        """Claim the paper for review: permanently delist it and lock the price."""
        if not self.can_start_review(agent):
            return False
        self.review_in_progress_by = agent
        self.review_claimed = True
        self.market_listed = False
        self.agreed_review_share = self.offered_share(agent)
        return True

    def finish_review(
        self,
        agent: Agent,
        effort: float,
        review_kind: str | None = None,
    ) -> float:
        """Finalize the in-progress review, granting the locked share.

        Returns the share actually transferred (0 below the effort threshold).
        Consumes the paper's single review either way.
        """
        if self.review_in_progress_by is not agent:
            return 0.0

        review_effort = self._nonnegative_float(effort, "review_effort")
        completed_review_kind = (
            normalize_review_kind(review_kind)
            if review_kind is not None
            else review_kind_from_effort(review_effort)
        )
        self.review_in_progress_by = None
        self.reviewed = True
        self.reviewer = agent

        share = 0.0
        epsilon = review_epsilon_from_effort(review_effort, self.quality)
        if review_effort >= MIN_REVIEW_EFFORT_THRESHOLD:
            share = min(
                self.agreed_review_share,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
            if share > 0.0:
                self.share_distribution[self.author] = (
                    self.share_distribution.get(self.author, 0.0) - share
                )
                self.share_distribution[agent] = (
                    self.share_distribution.get(agent, 0.0) + share
                )
            self.accrual_rate = self.estimate_accrual_rate_after_review(review_effort)
        self.review_records.append(
            {
                "reviewer": agent,
                "share": share,
                "effort": review_effort,
                "epsilon": epsilon,
                "review_kind": completed_review_kind,
                "accrual_rate": self.accrual_rate,
            }
        )
        return share

    def add_review(
        self,
        agent: Agent,
        effort: float,
        share: float | None = None,
    ) -> float:
        """Claim and finish a review in one call (direct/testing convenience)."""
        if not self.start_review(agent):
            return 0.0
        if share is not None:
            self.agreed_review_share = self._validate_share_value(share, "share")
        elif self.agreed_review_share <= 0.0:
            self.agreed_review_share = min(
                DEFAULT_REVIEW_SHARE,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
        return self.finish_review(agent, effort)

    def estimate_review_share(self, agent: Agent) -> float:
        """Share the author would grant ``agent`` for completing a review now."""
        if self.review_in_progress_by is agent:
            return min(
                self.agreed_review_share,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
        if not self.can_start_review(agent):
            return 0.0
        return min(
            self.offered_share(agent),
            max(0.0, self.share_distribution.get(self.author, 0.0)),
        )

    def estimate_accrual_rate_after_review(self, effort: float) -> float:
        return self.accrual_rate * (1.0 + review_accrual_bump(effort, self.quality))

    # ---- shares / accrual ----------------------------------------------
    def set_share(self, agent: Agent, share: float):
        if agent is None:
            raise ValueError("agent cannot be None")

        share_value = self._validate_share_value(share, "share")
        other_total = sum(
            current_share
            for contributor, current_share in self.share_distribution.items()
            if contributor is not agent
        )
        if other_total + share_value > 1.0 + 1e-12:
            raise ValueError("total paper shares cannot exceed 1.0")

        self.share_distribution[agent] = share_value

    def advance_accrual(self, time_steps: int = 1):
        self.accrue_ac(time_steps)

    def accrue_ac(self, time_steps: float = 1.0) -> float:
        """Increase current AC using the current provisional accrual rate."""
        elapsed = self._nonnegative_float(time_steps, "time_steps")
        self.current_ac += self.accrual_rate * elapsed
        return self.current_ac

    # ---- validation helpers --------------------------------------------
    def _validate_share_distribution(self):
        total = 0.0
        for contributor, share in self.share_distribution.items():
            if contributor is None:
                raise ValueError("share_distribution cannot contain None contributors")
            total += self._validate_share_value(share, "share_distribution share")
        if total > 1.0 + 1e-12:
            raise ValueError("initial share_distribution cannot exceed 1.0 total")

    @staticmethod
    def _nonnegative_float(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        return number

    @staticmethod
    def _validate_share_value(value: float, name: str) -> float:
        number = Paper._nonnegative_float(value, name)
        if number > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return number
