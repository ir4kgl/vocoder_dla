import torch


class FMLoss():
    def __call__(self, maps_gt, maps_pred):
        fm_loss = 0.
        for i in range(len(maps_gt)):
            for j in range(len(maps_gt[i])):
                fm_loss += torch.mean(
                    torch.abs(maps_gt[i][j]- maps_pred[i][j]))
        return fm_loss
