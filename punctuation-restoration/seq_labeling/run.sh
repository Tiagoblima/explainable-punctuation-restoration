# Copyright 2020 The HuggingFace Team. All rights reserved.
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
huggingface-cli login
HF_TOKEN='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'
python3 run_train.py \
  --model_name_or_path neuralmind/bert-base-portuguese-cased \
  --dataset_name tiagoblima/punctuation-nilc \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval \
  --do_predict \
  --push_to_hub \
  --push_to_hub_token $HF_TOKEN \
  --use_auth_token \
  --num_train_epochs 0.5 \
  --report_to wandb
