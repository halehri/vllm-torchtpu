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
"""Base TPU Model Runner for Omni models."""

from typing import Any

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class OmniTPUModelRunner(TPUModelRunner):
    """Shared base for every Omni Runner on TPU."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        if hasattr(self, "init_omni_connectors") and hasattr(self, "model_config"):
            self.init_omni_connectors(self.model_config)

    def _update_states(self, scheduler_output: Any) -> Any:
        if getattr(self, "_omni_connector_initialized", False):
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                self.cleanup_finished_request(req_id)
        return super()._update_states(scheduler_output)
