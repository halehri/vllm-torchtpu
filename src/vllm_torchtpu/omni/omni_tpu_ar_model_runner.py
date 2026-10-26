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
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner


class OmniTPUARModelRunner(OmniConnectorModelRunnerMixin, OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models."""

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        output = super().sample_tokens(grammar_output)
        if getattr(self, "_omni_connector_initialized", False):
            return self.attach_omni_connector_output(output)
        return output
