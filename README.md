# Flux Reasoner Engine

Dual-interpreter gradient reasoning engine. Two AI interpreters work in parallel on the same input; a gradient signal decides the output.

## Architecture

```
Input → [Creative Interpreter (Seed-2.0-mini)] ─┐
       [Logical Interpreter (DeepSeek-v4-flash)] ─┴─→ Gradient Gate → Decision
```

- **Creative Interpreter**: Generates N divergent options (high temperature, 0.85)
- **Logical Interpreter**: Evaluates against constraints (low temperature, 0.3)
- **Gradient**: `novelty - constraint` — measures how much the creative output breaks new ground vs. how much the logical evaluation contained it
- **Decision Threshold**: ~0.35 — above it, adopt creative; below half threshold, adopt logical; between, hold

## Installation

```bash
pip install flux-reasoner
```

## Usage

### Basic Usage

```python
from flux_reasoner import FluxReasoner

reasoner = FluxReasoner()
result = reasoner.reason(
    input="should we use async actors in holodeck-rust?",
    creative_prompt_template="Generate N divergent options for: {input}",
    logical_prompt_template="Evaluate critically and find flaws in: {input}",
    threshold=0.35
)
print(result["decision"])  # ADOPT_CREATIVE, ADOPT_LOGICAL, or HOLD
print(result["gradient"])  # 0.0 - 1.0
```

### Iterative Reasoning

```python
result = reasoner.reason_with_iterations(
    input="design a new caching strategy for distributed systems",
    iterations=3,
    threshold=0.35
)
print(result["converged"])  # True if gradient exceeded threshold
print(result["iterations"])  # List of iteration results
```

## API Reference

### FluxReasoner

#### `__init__(deepinfra_key: str = None)`
Initialize with optional DeepInfra API key. Falls back to environment variable `DEEPINFRA_API_KEY`.

#### `call_deepinfra_seed_mini(prompt: str, temperature: float = 0.85) -> str`
Call the Seed-2.0-mini model for creative divergent generation.

#### `call_deepseek(prompt: str) -> str`
Call DeepSeek-v4-flash via SiliconFlow for logical evaluation.

#### `compute_gradient(creative_output: str, logical_output: str) -> float`
Compute gradient = novelty - constraint. Returns value clamped to [0.0, 1.0].

#### `reason(input: str, creative_prompt_template: str, logical_prompt_template: str, threshold: float = 0.35) -> Dict`
Run single-pass dual-interpreter reasoning.

#### `reason_with_iterations(input: str, iterations: int = 3, threshold: float = 0.35) -> Dict`
Run iterative reasoning with creative ↔ logical feedback loops.

### Return Value

```python
{
    "input": str,           # Original input
    "creative_output": str, # Output from creative interpreter
    "logical_output": str,  # Output from logical interpreter
    "gradient": float,      # Computed gradient (0.0 - 1.0)
    "decision": str,        # ADOPT_CREATIVE | ADOPT_LOGICAL | HOLD
    "threshold": float      # Threshold used
}
```

## Gradient Signal

The gradient is computed as:

```
novelty = len(creative_words) / 50.0
constraint = len(intersection) / len(creative_words)
gradient = novelty - (constraint * 0.5)
```

- **novelty**: How many unique words did the creative interpreter produce?
- **constraint**: How many of those words overlap with the logical evaluation?
- **gradient**: Positive when creative breaks new ground; negative when logical constrained it

## Environment Variables

- `DEEPINFRA_API_KEY`: DeepInfra API key for Seed-2.0-mini
- `SILICONFLOW_KEY`: SiliconFlow API key for DeepSeek-v4-flash

## License

MIT
