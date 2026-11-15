"""Separate score-based promotion from the training objective."""
from __future__ import annotations
from dataclasses import dataclass, field
import math


def wilson_interval(wins, games, z=1.96):
    if not games: return (0.0, 1.0)
    p = wins/games; den = 1+z*z/games
    center = (p+z*z/(2*games))/den
    radius = z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/den
    return max(0,center-radius), min(1,center+radius)


@dataclass
class ProgressGuard:
    patience: int = 5
    history: list = field(default_factory=list)
    champion: dict | None = None
    stale: int = 0

    def observe(self, report):
        """Report must contain actual score metrics, not rewards-as-goal proxies."""
        required = ('score', 'goal_difference', 'cluster_fraction', 'loop_fraction', 'by_format')
        if any(k not in report for k in required): raise ValueError('incomplete independent evaluation')
        if not all(math.isfinite(float(report[k])) for k in required[:-1]):
            return {'promote':False,'reason':'nonfinite-evaluation','intervention':'rollback'}
        reward = float(report.get('training_reward',0))
        last = self.history[-1] if self.history else None
        hacking = bool(last and reward > last.get('training_reward',0)+max(.0001,abs(last.get('training_reward',0))*.1)
                       and report['goal_difference'] < last['goal_difference']-.15)
        bad_loop = report['loop_fraction'] > .45
        regression = False
        if self.champion:
            # A large-mode regression cannot hide behind strong duel results.
            for name, values in report['by_format'].items():
                old = self.champion['by_format'].get(name)
                if old and values['score'] < old['score']-.12:
                    regression = True
        improve = self.champion is None or report['score'] > self.champion['score']+.005
        promote = improve and not (hacking or bad_loop or regression)
        self.stale = 0 if promote else self.stale+1
        intervention = 'none'
        reason = 'improved-heldout-score' if promote else 'no-verified-improvement'
        if hacking or bad_loop:
            intervention='rollback-and-reduce-style';reason='reward-score-divergence' if hacking else 'loop-regression'
        elif regression:
            intervention='rebalance-regressed-formats';reason='format-regression'
        elif self.stale >= self.patience:
            intervention='refresh-opponents-and-curriculum';reason='plateau'
        self.history.append(dict(report));self.history=self.history[-32:]
        if promote: self.champion=dict(report)
        return {'promote':promote,'reason':reason,'intervention':intervention,'stale_evaluations':self.stale}
