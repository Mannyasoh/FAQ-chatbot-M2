import json
import os

DEFAULTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.6},
    "gpt-4o": {"input": 2.5, "cached_input": 1.25, "output": 10.0},
    "gpt-4.1-mini": {"input": 0.4, "cached_input": 0.1, "output": 1.6},
    "gpt-4.1": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.1, "output": 0.0},
}
PRICING = DEFAULTS.copy()
_env = os.getenv("OPENAI_PRICING_JSON")
if _env:
    try:
        override = json.loads(_env)
        for k, v in override.items():
            if isinstance(v, dict) and "input" in v:
                nv = {"input": float(v["input"]), "output": float(v.get("output", 0.0))}
                if "cached_input" in v:
                    nv["cached_input"] = float(v["cached_input"])
                PRICING[k] = nv
    except (ValueError, TypeError, KeyError):
        pass


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> float:
    if prompt_tokens < 0 or completion_tokens < 0 or cached_input_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    p = PRICING.get(model) or PRICING.get("gpt-4o-mini")
    if not p:
        raise ValueError(f"No pricing information available for model: {model}")
    try:
        input_rate = float(p["input"])
        output_rate = float(p.get("output", 0.0))
        cached_rate = float(p.get("cached_input", input_rate))
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid pricing data for model {model}: {e}") from e
    cached = max(0, int(cached_input_tokens))
    uncached = max(0, int(prompt_tokens) - cached)
    inp_cost = uncached / 1000000.0 * input_rate
    cached_cost = cached / 1000000.0 * cached_rate
    out_cost = completion_tokens / 1000000.0 * output_rate
    return round(inp_cost + cached_cost + out_cost, 6)


def estimate_embedding_cost(model: str, tokens: int) -> float:
    return estimate_cost_usd(model, tokens, 0, 0)
