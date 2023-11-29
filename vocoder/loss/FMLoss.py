import torch


class FMLoss():
    def __call__(self, maps_gt, maps_pred):
        fm_loss = 0.
        for i in range(len(maps_gt)):
            fm_loss += torch.mean(
                torch.stack(maps_gt[i]) - torch.stack(maps_pred[i]))
        return fm_loss
