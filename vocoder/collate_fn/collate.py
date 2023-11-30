import logging
from typing import List

import torch
from torch.nn import ConstantPad2d
from torch.nn.functional import pad

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["audio_path"] = list(x["audio_path"] for x in dataset_items)

    result_batch["audio"] = list(x["audio"] for x in dataset_items)
    result_batch["mel"] = list(x["mel"] for x in dataset_items)


    result_batch["audio_len"] = torch.as_tensor(
        list(x["audio"].shape[-1] for x in dataset_items), dtype=torch.int32)

    result_batch["mel_len"] = torch.as_tensor(
        list(x["mel"].shape[-1] for x in dataset_items), dtype=torch.int32)

    batch_audio_len = max(result_batch["audio_len"])
    batch_mel_len = max(result_batch["mel_len"])

    result_batch["audio"] = torch.cat(
        tuple(pad(x["audio"], (0, batch_audio_len - x["audio"].shape[-1]), "constant") for x in dataset_items)
    ).unsqueeze(1)

    result_batch["mel"] = torch.cat(
        tuple(pad(x["mel"], (0, batch_mel_len - x["mel"].shape[-1]), "constant") for x in dataset_items)
    )

    return result_batch
