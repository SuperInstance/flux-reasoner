"""
Flux Reasoner Engine — dual-interpreter gradient reasoning

Two interpreters, one input, gradient decides the output.

Creative interpreter (Seed-2.0-mini via DeepInfra): generates N divergent options
Logical interpreter (DeepSeek-v4-flash): evaluates against constraints

PLATO bridges them. The gradient = novelty - constraint.
Target gradient: ~0.35
"""

import os
import time
import requests
import json
from typing import List, Dict, Any, Optional, Callable

# DeepInfra config
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
DEEPINFRA_KEY = os.environ.get("DEEPINFRA_API_KEY", "RhZPtvuy4cXzu02LbBSffbXeqs5Yf2IZ")


class FluxReasoner:
    """
    Dual-interpreter reasoner with gradient control.

    Usage:
        reasoner = FluxReasoner()
        result = reasoner.reason(
            input="should we use async actors in holodeck-rust?",
            creative_prompt_template="Generate N divergent options for: {input}",
            logical_prompt_template="Evaluate critically: {input}",
            threshold=0.35
        )
        print(result["decision"])  # ADOPT_CREATIVE, ADOPT_LOGICAL, HOLD
    """

    def __init__(self, deepinfra_key: str = DEEPINFRA_KEY):
        self.deepinfra_key = deepinfra_key
        self.headers = {
            "Authorization": f"Bearer {deepinfra_key}",
            "Content-Type": "application/json"
        }

    def call_deepinfra_seed_mini(self, prompt: str, temperature: float = 0.85) -> str:
        """Call Seed-2.0-mini for creative divergent generation."""
        payload = {
            "model": "ByteDance/Seed-2.0-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 500
        }
        response = requests.post(
            f"{DEEPINFRA_BASE}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek-v4-flash for logical evaluation."""
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }
        response = requests.post(
            "https://api.siliconflow.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('SILICONFLOW_KEY', '')}"},
            json=payload,
            timeout=30
        )
        return response.json()["choices"][0]["message"]["content"]

    def call_glm(self, prompt: str, model: str = "glm-4.7") -> str:
        """Call GLM via OpenClaw sessions."""
        # This would integrate with OpenClaw's model routing
        return f"[GLM would evaluate: {prompt[:100]}...]"

    def compute_gradient(self, creative_output: str, logical_output: str) -> float:
        """
        Compute gradient = novelty - constraint.

        novelty: how divergent is the creative output?
        constraint: how much did the logical output limit it?
        """
        # Simple heuristic: count novel bigrams in creative vs constraint score in logical
        creative_words = set(creative_output.lower().split())
        logical_words = set(logical_output.lower().split())

        novelty = len(creative_words) / 50.0  # normalized
        constraint = len(logical_words & creative_words) / max(len(creative_words), 1)

        gradient = novelty - (constraint * 0.5)
        return min(max(gradient, 0.0), 1.0)

    def reason(
        self,
        input: str,
        creative_prompt_template: str = "Generate N divergent options for: {input}",
        logical_prompt_template: str = "Evaluate critically and find flaws in: {input}",
        threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Run dual-interpreter reasoning on an input.
        """
        creative_prompt = creative_prompt_template.format(input=input)
        logical_prompt = logical_prompt_template.format(input=input)

        # Run both interpreters
        creative_output = self.call_deepinfra_seed_mini(creative_prompt)
        time.sleep(0.5)  # rate limit
        logical_output = self.call_deepseek(logical_prompt)

        # Compute gradient
        gradient = self.compute_gradient(creative_output, logical_output)

        # Decide
        if gradient > threshold:
            decision = "ADOPT_CREATIVE"
        elif gradient < threshold * 0.5:
            decision = "ADOPT_LOGICAL"
        else:
            decision = "HOLD"

        return {
            "input": input,
            "creative_output": creative_output,
            "logical_output": logical_output,
            "gradient": gradient,
            "decision": decision,
            "threshold": threshold
        }

    def reason_with_iterations(
        self,
        input: str,
        iterations: int = 3,
        threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Run iterative reasoning: creative → logical → creative → logical...
        Each iteration, the creative output is refined based on logical feedback.
        """
        results = []
        creative_output = None
        logical_output = None

        for i in range(iterations):
            if i == 0:
                creative_prompt = f"Generate N divergent, creative options for: {input}"
            else:
                creative_prompt = f"Refine these options based on the critique: {creative_output[:500]}\n\nCritique: {logical_output[:500]}\n\nGenerate improved options:"

            logical_prompt = f"Critically evaluate and find the strongest flaws in: {creative_output or input}"

            creative_output = self.call_deepinfra_seed_mini(creative_prompt)
            time.sleep(0.5)
            logical_output = self.call_deepseek(logical_prompt)

            gradient = self.compute_gradient(creative_output, logical_output)
            results.append({
                "iteration": i + 1,
                "creative": creative_output,
                "logical": logical_output,
                "gradient": gradient
            })

            if gradient > threshold:
                break

        return {
            "input": input,
            "iterations": results,
            "final_gradient": results[-1]["gradient"],
            "converged": results[-1]["gradient"] > threshold
        }
