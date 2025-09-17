set -x

ray stop --force

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
N_GPUS_PER_NODE=4 # default is 4

PROJ_ROOT=/inspire/ocean/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/liushan/third_party/verl
HF_MODEL_PATH=/inspire/ocean/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/liushan/models/Qwen/Qwen3-0.6B
DIST_CKPT_PATH=${PROJ_ROOT}/checkpoint/qwen3_06b_torch_dist

# convert HF model to meagatron format offlinely
# python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH


# megatron tuning guide:
# 1. recommend to offload all states by setting ALL_OFFLOAD=True
# 2. enable dynamic batch size by setting actor_rollout_ref.actor.use_dynamic_bsz=True ref.log_prob_use_dynamic_bsz=True rollout.log_prob_use_dynamic_bsz=True
# 3. set ppo_max_token_len_per_gpu and log_prob_max_token_len_per_gpu as large as possible for better MFU (limited by GPU memory). assure ppo_max_token_len_per_gpu > max_prompt_length+max_response_length, if sequence length is too long, you can increase the TP/PP size
# 4. if memory is very limited, enable full recompute, but the mfu will be 30% lower
#        full recompute settings:
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
#        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}


train_path=$PROJ_ROOT/dataset/gsm8k/train.parquet
test_path=$PROJ_ROOT/dataset/gsm8k/test.parquet

python3 -m verl.trainer.mlm_sft_trainer --config-path=config \
    --config-name='mlm_sft_trainer.yaml'\
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_length=1024 \
    data.truncation='error' \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.entropy_coeff=1.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    trainer.logger='["console"]' \
    trainer.project_name='verl_sft_example' \
    trainer.experiment_name='qwen2_5_vl_3b_megatron' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 $@
