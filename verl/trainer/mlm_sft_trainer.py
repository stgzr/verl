"""
"""
import os
import re
import logging
from functools import partial
from typing import Iterable
from collections import defaultdict
import socket
from typing import Optional
from pprint import pprint
import hydra
import ray
import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.device import is_cuda_available, get_device_name, get_device_id
from verl.utils.debug import marked_timer
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    process_validation_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.megatron_workers import ActorRolloutRefWorker
from verl.utils.profiler import log_gpu_memory_usage, GPUMemoryLogger, DistProfiler
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.device import get_torch_device
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.dataset.sft_dataset import SFTDataset


WorkerType = type(Worker)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronSFTDataset(SFTDataset):
    def __init__(self,
                 parquet_files: str | ListConfig,
                 tokenizer: PreTrainedTokenizer,
                 config: DictConfig,
                 processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(parquet_files, tokenizer, config)
        self.processor = processor
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()

        """
        # NOTE: remap dataframe to Dataset format.

        from datasets import Dataset
        import pandas as pd
        breakpoint()
        if isinstance(self.dataframe, pd.DataFrame):
            self.dataframe = Dataset.from_pandas(self.dataframe)

        """

        # NOTE: Since `pd.read_parquet` would cheat PIL.Image as bytes,
        #       here we rebuild self.dataframe with `datasets.load_dataset` instead.
        import datasets
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        # TODO: `maybe_filter_out_long_prompts`?

    def __getitem__(self, item):
        data_item = super().__getitem__(item)
        input_ids = data_item["input_ids"]
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        labels[-1] = self.tokenizer.pad_token_id
        data_item["labels"] = labels
        data_item["indice"] = item

        row_dict = self.dataframe[item]
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use image here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            # TODO: add multi-modal inputs here
            raw_prompt = self.prompts[item]
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            # TODO: multimodal training with data packing? or single sample training

        return data_item


class ActorRolloutRefWorkerMlmSFTTrainer(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            
            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        assert self._is_actor, f"You must set actor in {self.__class__.__name__}"

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()),
            resolve=True)

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        # we need the model for actor
        optim_config = self.config.actor.optim
        (
            self.actor_module,
            self.actor_optimizer,
            self.actor_optimizer_scheduler,
            self.actor_model_config,
            self.actor_optim_config,
        ) = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        OmegaConf.set_struct(self.config.actor, True)
        with open_dict(self.config.actor):
            use_fused_kernels = self.config.model.get("use_fused_kernels", False)
            self.config.actor.use_fused_kernels = use_fused_kernels

        actor_cls = MegatronSFTActor
        self.actor = actor_cls(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage(f"After {type(actor_cls).__name__} init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.actor.checkpoint,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.actor_optimizer,
            optimizer_scheduler=self.actor_optimizer_scheduler,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
        )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

        self.eval_cnt = 0

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)  # NOTE: it will chunk data-batch
    @GPUMemoryLogger(role="evaluation", logger=logger)
    @DistProfiler.annotate(color="red")
    def evaluate(self, data: DataProto):
        # if torch.distributed.get_rank() in (0, 1):
        #     breakpoint()
        self.eval_cnt += 1
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)

        data.batch = data.batch.to(get_device_name())

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data, eval_mode=True)

        with Timer(name="evaluate", logger=None) as timer:
            # breakpoint()
            metrics = self.actor.evaluate(dataloader=dataloader)

        if torch.distributed.get_rank() in (2,3):
            # NOTE: no collect_fn called?
            logger.warning(f" >>>>>>>>> [{torch.distributed.get_rank()}] (eval-iter = {self.eval_cnt}) get_metrics: {metrics}")
            import sys, time
            time.sleep(5)
            # sys.exit(1)
        # delta_time = timer.last
        # global_num_tokens = data.meta_info["global_token_num"]

        # output = DataProto(meta_info={"metrics": metrics})
        output = DataProto(non_tensor_batch={"loss": metrics})  # NOTE: meta_info might lose in `collect_fn`
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        get_torch_device().empty_cache()
        return output


class MegatronSFTActor(MegatronPPOActor):
    """Megatron SFT trainer
    """
    def make_minibatch_iterator(self, data: DataProto, eval_mode: bool = False) -> Iterable[DataProto]:
        """Assume data is a global batch
        """

        # NOTE: shard data via dp_size first
        """
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()

        if not eval_mode:
            data_shard = data.chunk(dp_size)[dp_rank]
        else:
            # NOTE: Since you might do evaluation via an indivisible chunk pattern,
            #       as `drop_last=False` in dataloader,
            #       so we use split method to split batch data instead.
            if torch.distributed.get_rank() == 0:
                # breakpoint()
                pass
            remaining_batch_size = data.batch.batch_size.numel() % dp_size
            split_size = data.batch.batch_size.numel() // dp_size + int(remaining_batch_size > 0)
            data_shard = data.split(split_size)[dp_rank]
        """
        data_shard = data

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "loss_mask",
            "labels"
        ]
        self.has_multi_modal_inputs = "multi_modal_inputs" in data_shard.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            data_shard = data_shard.select(select_keys, ["multi_modal_inputs"])
        else:
            data_shard = data_shard.select(batch_keys=select_keys)
        return data_shard.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,  # TODO: change to another config?
            epochs=self.config.ppo_epochs,  # TODO
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle}, 
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None, 
        mini_batch_size=None,
    ):
        """
        """
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)
    
        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward
        
        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            device = output["log_probs"].device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            loss_mask = data["loss_mask"]
            response_mask = data["loss_mask"].to(bool)
            loss_agg_mode = self.config.loss_agg_mode

            # compute loss
            ret_entropy = None
            stats = {}
            if not forward_only:
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

            if calculate_entropy:
                entropy = output["log_probs"]
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = entropy_coeff * entropy_loss
                    metrics["actor/entropy loss"] = entropy_loss.detach().item()
                else:
                    ret_entropy = entropy
            
            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    # TODO: not implemented yet
                    pass
            
            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]
        
        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                for key in batch["multi_modal_inputs"][0].keys():
                    idxs = batch["multi_modal_inputs_idx"]
                    mmi = batch["multi_modal_inputs"]
                    multi_modal_inputs[key] = torch.cat(
                        [mmi[idx].get(key) for idx in idxs if mmi[idx].get(key) is not None], dim=0
                    )

            label = batch["labels"]
            label_mask = batch["loss_mask"]

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                # return dict of [logits, entropy]
                output = forward_fn(
                    model,
                    input_ids,
                    position_ids,
                    attention_mask,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    labels=label,
                    labels_maks=label_mask,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    ret = {}

                    # NOTE: below cause inplace gradient modification error ...
                    log_probs = -vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask.bool(), 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    sequence_parallel=self.tf_config.sequence_parallel,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                )
        
            if forward_only:
                meta_info = None
            else:
                meta_info = {"entropy_coeff": self.config.entropy_coeff,}

            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batchs inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced 

    def evaluate(self, dataloader: Iterable[DataProto]) -> dict:
        metrics = []
        self.prof.start()

        # Turn on evaluation mode
        for model_module in self.actor_module:
            model_module.eval()

        for data in dataloader:
            data.to(get_device_id())
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size

            with torch.no_grad():
                metric_micro_batch = self.forward_backward_batch(
                    data,
                    calculate_entropy=(self.config.entropy_coeff != 0),
                    use_dynamic_bsz=self.config.use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                    mini_batch_size=self.config.ppo_mini_batch_size,
                )
                metrics.append(metric_micro_batch)

        # Turn off evaluation mode
        for model_module in self.actor_module:
            model_module.train()

        def collect_metrics(metrics, key):
            loss_items = []
            outputs = [metric["output"] for metric in metrics]
            for output in outputs:
                for item in output:
                    loss_item = item[0][key]
                    loss_items.append(loss_item)
            return loss_items

        loss_collect = collect_metrics(metrics, key="actor/entropy loss")
        loss = np.array(loss_collect)

        return loss


def compute_data_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics.
    """
    metrics = {}

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    return metrics


# TODO: configure file for this application
@hydra.main(config_path="config", config_name="mlm_sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def run_sft(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus)

    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filenam=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:

    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # TODO: you need to change `actor_rollout_ref` to `actor`
        assert config.actor_rollout_ref.actor.strategy in ("megatron", "megatron.sft")
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        role_worker_mapping = {
            Role.Actor: ray.remote(ActorRolloutRefWorkerMlmSFTTrainer),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.Actor: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        train_dataset, val_dataset, train_sampler = None, None, None

        trainer = RaySFTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=NVMegatronRayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


class RaySFTTrainer:
    """Distributed SFT trainer using Ray for resource scheduling.

    This trainer orchestrates distributed SFT training across multiple nodes and GPUs,
    managing actor training with Ray backend.
    Supports Megatron integration.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed SFT trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self._validate_config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = train_sampler
        # TODO: how to set rank and world_size for distributed-sampler
        self._create_dataloader(self.train_dataset, self.val_dataset, self.train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # TODO: replace `actor_rollout_ref` to `actor`
        assert config.actor_rollout_ref.actor.strategy in ("megatron", "megatron.sft")
        if config.actor_rollout_ref.actor.strategy in ("megatron", "megatron.sft"):
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * 1 # config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"


        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        if self.config.data.custom_cls.get("path", None):
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(self.config.data.custom_cls.path,
                                           self.config.data.custom_cls.name)
        elif self.config.data.get("multiturn", {}).get("enable", False):
            # TODO: seems not support yet!
            from verl.trainer.fsdp_sft_trainer import MultiTurnSFTDataset
            dataset_cls = MultiTurnSFTDataset
        else:
            # from verl.utils.dataset.sft_dataset import SFTDataset
            dataset_cls = MegatronSFTDataset # SFTDataset # MegatronSFTDataset
        
        train_dataset = dataset_cls(parquet_files=self.config.data.train_files, tokenizer=self.tokenizer, config=self.config.data, processor=self.processor)
        val_dataset = dataset_cls(parquet_files=self.config.data.val_files, tokenizer=self.tokenizer, config=self.config.data, processor=self.processor)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # TODO: need to process SP or CP?

        if train_sampler is None:
            # NOTE: Here, we treat the global batch-size collected in the single-controller,
            # TODO: you should set the seed of data sampler.
            train_sampler = torch.utils.data.RandomSampler(
                self.train_dataset)
        
        num_workers = self.config.data["dataloader_num_workers"]
        # TODO: batch_size config
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,  # The global batchsize
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
            pin_memory=True
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = self.config.data.train_batch_size

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=True,  # NOTE: In distributed valiation, drop_last=False is not supported! 
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _validate(self):
        # TODO!!!: rewrite validation for SFT training
        # TODO: shrink processing
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                # non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            # TODO:
            # first chunk batch
            # then make iterator
            # forward-only mode
            # accumulate loss
            test_batch = test_batch.union(test_gen_batch)
            # breakpoint()

            # if test_batch.batch.batch_size.numel() % self.actor_wg.dp_size != 0:
            #     nearest_batch_size = test_batch.batch.batch_size.numel() // self.actor_wg.dp_size * self.actor_wg.dp_size
            #     test_batch = test_batch.split(nearest_batch_size)[0]


            # print(f"new_test_batch: {test_batch.batch.batch_size.numel()}")

            # actor_wg:
            breakpoint()
            score = self.actor_wg.evaluate(test_batch)
            score = score.non_tensor_batch['loss'].mean().item()  # NOTE: forward-only mode could not log total_num_tokens

            print(f"~~~~~ score: {score}")

            # NOTE: 
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "validate": True,
                "global_steps": self.global_steps,
            }
            # print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # Store generated outputs
            test_batch.meta_info["validate"] = True

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        data_sources = ["test_source"]  # np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for Actor.
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref, # NOTE: change to actor configure
            role="actor",
            profile_option=self.config.trainer.npu_profile.options
        )
        self.resource_pool_to_cls[resource_pool]['actor'] = actor_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # TODO: you do not need to use colocated setting, since you are SFTing
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        
        self.actor_wg = all_wg["actor"]
        self.actor_wg.init_model()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        # load actor
        self.actor_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.start_profile(role="e2e", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        torch.autograd.set_detect_anomaly(True)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # NOTE: batch_dict contains dp_size * micro_batch?

                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = [
                    "input_ids", "attention_mask", "position_ids", "loss_mask",
                    # "responses", "response_mask"
                ]
                non_tensor_batch_keys_to_pop = [] #["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                batch = batch.union(gen_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                # gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    # TODO: to match this features
                    if False and self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # breakpoint()
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # validate
                    if (
                        # self.val_reward_fn is not None and
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


if __name__ == "__main__":
    main()
