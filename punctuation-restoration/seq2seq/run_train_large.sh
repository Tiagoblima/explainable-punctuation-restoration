
#huggingface-cli login
HF_TOKEN='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'
python3 run_train.py \
  --model_name_or_path unicamp-dl/ptt5-large-portuguese-vocab \
  --dataset_name tiagoblima/punctuation-tedtalk2012-t5 \
  --output_dir tiagoblima/punctuation-tedtalk2012-t5-large \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --push_to_hub \
  --num_train_epochs 1 \
  --push_to_hub_token $HF_TOKEN \
  --use_auth_token \
  --report_to wandb