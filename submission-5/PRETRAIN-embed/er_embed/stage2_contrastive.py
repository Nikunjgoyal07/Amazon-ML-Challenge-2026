
"""Stage 2 of 02c: contrastive training (one process per GPU under torchrun, or a single process)."""
import inspect
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback
try:                                             # sentence-transformers >= 6 moved these; older versions keep the old path
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers, MultiDatasetBatchSamplers
except ImportError:
    from sentence_transformers import losses
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
world = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("LOCAL_RANK", 0))
per_device = max(1, cfg["batch_size"] // world)
model = SentenceTransformer(cfg["init_model"], device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
model.max_seq_length = cfg["max_seq_length"]
data = {name: Dataset.from_pandas(pd.read_parquet(p), preserve_index=False) for name, p in cfg["train_files"].items()}
train_data = next(iter(data.values())) if len(data) == 1 else data

cached = per_device > cfg["cache_above"]
loss_cls = losses.CachedMultipleNegativesRankingLoss if cached else losses.MultipleNegativesRankingLoss
kwargs = {"scale": cfg["scale"]}
if cached:
    kwargs["mini_batch_size"] = cfg["mini_batch"]
gather = world > 1 and "gather_across_devices" in inspect.signature(loss_cls.__init__).parameters
if gather:
    kwargs["gather_across_devices"] = True
loss = loss_cls(model, **kwargs)
steps = sum(len(d) // cfg["batch_size"] for d in data.values()) * cfg["epochs"]

args = SentenceTransformerTrainingArguments(
    output_dir=cfg["checkpoint_dir"],
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=per_device,
    learning_rate=cfg["learning_rate"],
    warmup_steps=int(cfg["warmup_share"] * steps),
    fp16=cfg["fp16"],
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    dataloader_drop_last=True,
    save_strategy="no",
    logging_steps=cfg["logging_steps"],
    report_to="none",
    seed=cfg["seed"],
    disable_tqdm=True,
    ddp_timeout=600,                             # a stuck GPU link fails after 10 min instead of 30
)


class Progress(TrainerCallback):
    """Step, loss and time left at every log (first process only)."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and "loss" in logs:
            minutes = (time.time() - self.t0) / 60
            left = minutes / max(state.global_step, 1) * (state.max_steps - state.global_step)
            print(f"step {state.global_step:,}/{state.max_steps:,}  loss {logs['loss']:.4f}  "
                  f"lr {logs.get('learning_rate', 0):.2e}  {minutes:.1f} min, ~{left:.0f} min left", flush=True)


trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_data, loss=loss, callbacks=[Progress()])
trainer.remove_callback(PrinterCallback)
if trainer.is_world_process_zero():
    chunks = f" in chunks of {cfg['mini_batch']} rows" if cached else ""
    negatives = " gathered across GPUs" if gather else (" per GPU only (this version cannot gather)" if world > 1 else "")
    print(f"{world} process(es) x {per_device} rows per step | {loss_cls.__name__}{chunks} | in-batch negatives"
          f"{negatives} | pair files: " + ", ".join(f"{k} {len(v):,}" for k, v in data.items()), flush=True)
t0 = time.time()
trainer.train()
if trainer.is_world_process_zero():
    model.save(cfg["model_dir"])
    log = pd.DataFrame([r for r in trainer.state.log_history if "loss" in r])
    (log if len(log) else pd.DataFrame(columns=["step", "loss", "learning_rate"])).to_csv(cfg["log_file"], index=False)
    Path(cfg["done_file"]).write_text(json.dumps({
        "minutes": round((time.time() - t0) / 60, 1), "processes": world, "rows_per_gpu": per_device,
        "loss": loss_cls.__name__, "gather_across_gpus": gather}), encoding="utf-8")
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()                  # the other processes wait until the model is saved
    torch.distributed.destroy_process_group()
