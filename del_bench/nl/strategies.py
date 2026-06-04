"""Prompt strategies. Direct vs. CoT. Each builds the strategy-specific
suffix appended to the system prompt and the JSON schema for responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PromptStrategy:
    name: str
    system_suffix: str
    json_schema_extra_props: tuple = ()

    def schema_for_choices(self, labels: Tuple[str, ...]) -> dict:
        props: dict = {
            "answer": {"type": "string", "enum": list(labels)},
        }
        required = ["answer"]
        for prop_name, prop_spec in self.json_schema_extra_props:
            props[prop_name] = prop_spec
            required.append(prop_name)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "probe_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


DIRECT = PromptStrategy(
    name="direct",
    system_suffix=(
        "Choose your answer by selecting the option whose label exactly matches "
        "your judgement. Reply with JSON only in the form "
        '{"answer": "<exact option label>"}.'
    ),
)


COT = PromptStrategy(
    name="cot",
    system_suffix=(
        "First think step by step about what each person knows or believes "
        "given the scenario. Identify whose perspective the question is asking "
        "about, then trace through the relevant beliefs. After your reasoning, "
        "give your final answer. Reply with JSON only in the form "
        '{"reasoning": "<your brief reasoning>", "answer": "<exact option label>"}.'
    ),
    json_schema_extra_props=(("reasoning", {"type": "string"}),),
)


STRATEGIES = {DIRECT.name: DIRECT, COT.name: COT}


def get(name: str) -> PromptStrategy:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}. available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
