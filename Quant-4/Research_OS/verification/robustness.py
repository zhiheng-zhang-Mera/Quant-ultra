"""Deterministic adversarial attack selection and evidence aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from Research_OS.contracts.verification import RobustnessEvidence


@dataclass(frozen=True)
class Attack:
    name: str
    critical: bool
    runner: Callable[[dict], dict]


class AttackRegistry:
    REQUIRED_NAMES = frozenset({
        "cost_multiplier", "execution_delay", "remove_best_year", "remove_best_period", "remove_top_contributors",
        "remove_top_sector", "parameter_perturbation", "rebalance_day_shift", "universe_perturbation",
        "liquidity_filter_change", "signal_randomization_placebo", "label_randomization_placebo",
        "warm_start_fold_checks", "capital_scale_sensitivity",
    })

    def __init__(self):
        self.attacks: dict[str, Attack] = {}

    def register(self, attack: Attack) -> None:
        if attack.name not in self.REQUIRED_NAMES:
            raise ValueError("unknown or ungoverned attack")
        if attack.name in self.attacks:
            raise ValueError("attack already registered")
        self.attacks[attack.name] = attack

    def run(self, experiment_id: str, selected: tuple[str, ...], primary: dict) -> RobustnessEvidence:
        if not selected:
            raise ValueError("attacks must be preregistered or policy-selected")
        unknown = set(selected) - set(self.attacks)
        if unknown:
            raise ValueError(f"unregistered attacks: {sorted(unknown)}")
        results, failures, mechanisms = {}, [], {}
        for name in selected:
            result = self.attacks[name].runner(dict(primary))
            if not isinstance(result, dict) or "passed" not in result:
                raise ValueError("attack runner returned malformed evidence")
            results[name] = result
            if self.attacks[name].critical and not result["passed"]:
                failures.append(name)
                mechanisms[name] = str(result.get("failure_mechanism", "unspecified robustness failure"))
        return RobustnessEvidence(schema_version="robustness-evidence/v1", experiment_id=experiment_id,
                                  attacks=results, critical_failures=tuple(failures), failure_mechanisms=mechanisms)


def placebo_attack(signal_key: str, alpha_key: str) -> Attack:
    def run(payload: dict) -> dict:
        randomized = bool(payload.get(signal_key + "_randomized", False))
        inherited_alpha = bool(payload.get(alpha_key, False))
        return {"passed": randomized and not inherited_alpha,
                "failure_mechanism": "randomized placebo inherited primary alpha" if inherited_alpha else ""}
    return Attack("signal_randomization_placebo", True, run)
