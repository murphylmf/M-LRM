import argparse
import os
import logging
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import signal
from lightning.pytorch.plugins.environments import SLURMEnvironment

import lrm
from lrm.datasets import ObjaverseDataModule
from lrm.utils.misc import get_rank
from lrm.utils.base import find_class
from lrm.utils.typing import Optional

def main(args, extras) -> None:
    logger = logging.getLogger("pytorch_lightning")
    
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(get_rank() + cfg.seed, workers=True)

    system = find_class(cfg.system_cls)(cfg.system)
    system.from_pretrained()
    # system = system.to(torch.device("cuda"))

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.system.save_dir, "checkpoints"),
            filename="model-{epoch:02d}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=args.save_every_n_epochs,
        ),
    ]

    wandb_logger = WandbLogger(
        project="MLRM",
        name=f"{args.name}",
        resume=True if args.resume is not None else False,
    )
    system._wandb_logger = wandb_logger
    loggers = [wandb_logger]
    if not args.wandb_upload:
        os.environ["WANDB_MODE"] = "offline"

    dm = ObjaverseDataModule(args.batch_size, args.batch_size_eval, args.num_workers)

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices="auto",
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.save_every_n_epochs,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp",
        log_every_n_steps=1,
    )

    if not os.path.exists(cfg.system.save_dir):
        os.makedirs(cfg.system.save_dir)

    trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
    trainer.test(system, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--name", type=str, default="MLRM_base", help="name of the experiment")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--wandb_upload", action="store_true", help="upload to wandb")

    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--max_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--num_sanity_val_steps", type=int, default=1, help="number of sanity validation steps")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="save every n epochs")
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="evaluate every n epochs")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--batch_size_eval", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="accumulate grad batches")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

    args, extras = parser.parse_known_args()

    main(args, extras)
