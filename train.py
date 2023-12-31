import argparse
import collections
import warnings

import numpy as np
import torch

import vocoder.loss as module_loss
import vocoder.model as module_arch
from vocoder.trainer import Trainer
from vocoder.utils import prepare_device
from vocoder.utils.object_loading import get_dataloaders
from vocoder.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config["arch_g"], module_arch)
    discriminator = config.init_obj(config["arch_d"], module_arch)
    logger.info(generator)
    logger.info(discriminator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    # get function handles of loss and metrics
    adv_loss_g = config.init_obj(config["adv_loss_g"], module_loss)
    adv_loss_d = config.init_obj(config["adv_loss_d"], module_loss)
    mel_loss = config.init_obj(config["mel_loss"], module_loss)
    fm_loss = config.init_obj(config["fm_loss"], module_loss)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_g = filter(lambda p: p.requires_grad, generator.parameters())
    trainable_params_d = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_g = config.init_obj(config["optimizer_g"], torch.optim, trainable_params_g)
    optimizer_d = config.init_obj(config["optimizer_d"], torch.optim, trainable_params_d)
    lr_scheduler_g = config.init_obj(config["lr_scheduler_g"], torch.optim.lr_scheduler, optimizer_g)
    lr_scheduler_d = config.init_obj(config["lr_scheduler_d"], torch.optim.lr_scheduler, optimizer_d)

    trainer = Trainer(
        generator,
        discriminator,
        adv_loss_g,
        adv_loss_d,
        mel_loss,
        fm_loss,
        optimizer_g,
        optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
