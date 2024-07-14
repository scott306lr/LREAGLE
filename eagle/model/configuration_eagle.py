# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""LLaMA model configuration"""

from transformers import PretrainedConfig

class EagleConfig(PretrainedConfig):
    model_type = "eagle"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        base_model_name_or_path="meta-llama/Llama-2-7b-chat-hf", # [MODIFIED] target model path
        **kwargs,
    ):

        self.base_model_name_or_path = base_model_name_or_path # [MODIFIED] target model path
        super().__init__(
            **kwargs,
        )

if __name__ == "__main__":
    eagle_config = EagleConfig()
    eagle_config.save_pretrained("eagle-llama2-7b")
    