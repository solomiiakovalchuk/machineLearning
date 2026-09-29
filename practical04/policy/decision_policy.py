"""Політика прийняття рішень для FNA-тріажу (Практична робота № 4).

Компонент перетворює бал моделі (оцінка ймовірності злоякісності у [0, 1])
на назву операційної дії. Числові параметри НЕ зберігаються в коді:
вони завантажуються з policy_config.json (див. DecisionPolicy.from_config).

Правило порівняння скрізь однакове: score >= threshold належить до зони вище порога.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

# Назви дій (єдине джерело; notebook записує їх у конфігурацію звідси)
ACTION_STANDARD = "standard_followup"   # планове спостереження
ACTION_REVIEW = "manual_review"         # повторна FNA + консиліум
ACTION_BIOPSY = "biopsy"                # core-needle біопсія + консультація онколога

DEFAULT_CONFIG_PATH = Path(__file__).with_name("policy_config.json")


def _check_score(score) -> float:
    """Перевіряє бал: число, скінченне, у діапазоні [0, 1]."""
    if isinstance(score, (bool, str, bytes)):
        raise TypeError(f"score must be a real number, got {type(score).__name__}")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"score must be a real number, got {score!r}") from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"score must be finite, got {value}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class DecisionPolicy:
    """Три зони за двома порогами.

    score <  threshold_low                    -> standard_followup
    threshold_low <= score < threshold_high   -> manual_review
    score >= threshold_high                   -> biopsy

    Однопорогова політика задається threshold_low == threshold_high:
    тоді зона ручного перегляду порожня.
    """

    threshold_low: float
    threshold_high: float
    action_low: str = ACTION_STANDARD
    action_mid: str = ACTION_REVIEW
    action_high: str = ACTION_BIOPSY

    def __post_init__(self) -> None:
        for name in ("threshold_low", "threshold_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if math.isnan(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.threshold_low > self.threshold_high:
            raise ValueError(
                "threshold_low must be <= threshold_high "
                f"(got {self.threshold_low} > {self.threshold_high})"
            )

    # ---- одне рішення -------------------------------------------------
    def decide(self, score: float) -> str:
        value = _check_score(score)
        if value >= self.threshold_high:
            return self.action_high
        if value >= self.threshold_low:
            return self.action_mid
        return self.action_low

    # ---- пакет: порядок результатів = порядок входу --------------------
    def decide_batch(self, scores: Iterable[float]) -> List[str]:
        return [self.decide(s) for s in scores]

    def decide_batch_with_quota(self, scores: Sequence[float], max_share: float) -> List[str]:
        """Пакетне рішення з жорсткою квотою на дію високої зони (biopsy).

        Спочатку застосовуються пороги. Якщо кількість біопсій > k = capacity_k(n, max_share),
        лишаються k об'єктів із найвищим балом (ties: менший індекс першим), решта
        переводяться на один рівень вниз -> manual_review (черга не губиться).
        """
        scores = list(scores)
        actions = self.decide_batch(scores)
        k = capacity_k(len(scores), max_share)
        biopsy_idx = [i for i, a in enumerate(actions) if a == self.action_high]
        if len(biopsy_idx) <= k:
            return actions
        keep = {biopsy_idx[j] for j in select_top_k([scores[i] for i in biopsy_idx], k)}
        for i in biopsy_idx:
            if i not in keep:
                actions[i] = self.action_mid
        return actions

    # ---- завантаження з конфігурації ----------------------------------
    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG_PATH, variant: str | None = None):
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        name = variant or cfg["active_variant"]
        params = cfg["policy_variants"][name]
        actions = cfg["actions"]
        return cls(
            threshold_low=params["threshold_low"],
            threshold_high=params["threshold_high"],
            action_low=actions["low"],
            action_mid=actions["middle"],
            action_high=actions["high"],
        )


def capacity_k(n_objects: int, max_share: float) -> int:
    """Кількість місць квоти: floor(max_share * n) (із запасом на похибку float)."""
    if n_objects < 0:
        raise ValueError("n_objects must be >= 0")
    if not 0.0 <= max_share <= 1.0:
        raise ValueError("max_share must be in [0, 1]")
    return int(math.floor(n_objects * max_share + 1e-9))


def select_top_k(scores: Sequence[float], k: int) -> List[int]:
    """Індекси k об'єктів із найвищим балом, у порядку спадання бала.

    Ties розв'язуються детерміновано: при рівних балах першим іде об'єкт
    із меншим індексом (порядок надходження). Якщо k >= len(scores),
    повертаються всі об'єкти (партія менша за k); порожня партія -> [].
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be >= 0")
    values = [_check_score(s) for s in scores]
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]
