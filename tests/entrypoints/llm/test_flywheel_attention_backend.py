# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""--attention-backend flywheel against the default RPA backend: greedy
Qwen3-0.6B generations on one TPU core.

Each engine runs in its own process (libtpu is process-exclusive), in process
with the engine core (VLLM_ENABLE_V1_MULTIPROCESSING=0), which starts no child
process for --attention-backend flywheel's runtime-registered name to reach.
The prompts outgrow max_num_batched_tokens, so the steps cover fresh and
chunked prefill, mixed prefill + decode, and decode alone.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.tpu_test

_GENERATE = """
import json, sys
from vllm import LLM, SamplingParams

backend = sys.argv[1] or None
base = "The quick brown fox jumps over the lazy dog. "
prompts = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "List the planets of the solar system in order:",
    "Explain why the sky is blue in one paragraph.",
    "def fibonacci(n):",
    base * 20 + "Summarize the text above in one sentence.",
    base * 45 + "How many times does the word fox appear above?",
    "1, 2, 3, 4, 5,",
    "Translate to German: The weather is nice today.",
    base * 70 + "What animal jumps in the text above?",
]
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=2048,
    max_num_batched_tokens=256,
    max_num_seqs=8,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    attention_backend=backend,
)
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
print("TOKEN_IDS=" + json.dumps([list(o.outputs[0].token_ids) for o in outputs]))
"""


def _generate(attention_backend: str | None) -> list[list[int]]:
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
    result = subprocess.run(
        [sys.executable, "-c", _GENERATE, attention_backend or ""],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line for line in result.stdout.splitlines() if line.startswith("TOKEN_IDS=")
    ]
    assert result.returncode == 0 and lines, (
        f"{attention_backend or 'default'} backend failed "
        f"(rc={result.returncode}):\n{result.stderr[-6000:]}"
    )
    return json.loads(lines[-1].removeprefix("TOKEN_IDS="))


def test_flywheel_generates_rpa_tokens():
    rpa = _generate(None)
    flywheel = _generate("flywheel")

    assert len(flywheel) == len(rpa)
    for i, (got, want) in enumerate(zip(flywheel, rpa)):
        common = next(
            (k for k, (a, b) in enumerate(zip(got, want)) if a != b), len(want)
        )
        # bf16 kernels with different reduction orders may part on a late
        # near-tie, never on the first tokens.
        assert common >= 16, f"prompt {i}: {got} vs {want}"
    exact = sum(got == want for got, want in zip(flywheel, rpa))
    assert exact >= len(rpa) - 2, f"{exact}/{len(rpa)} identical"
