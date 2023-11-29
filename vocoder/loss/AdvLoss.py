import torch


class AdversarialLossGenerator():
    def __call__(outputs_disc):
        adv_loss_gen = 0.
        for i in range(len(outputs_disc)):
            adv_loss_gen += torch.mean((1 - outputs_disc[i]) ** 2)
        return adv_loss_gen

class AdversarialLossDiscriminator():
    def __call__(outputs_gt, outputs_preds):
        adv_loss_disc = 0.
        for i in range(len(outputs_gt)):
            adv_loss_disc += torch.mean((1 - outputs_gt[i]) ** 2)
            adv_loss_disc += torch.mean(outputs_preds[i] ** 2)
        return adv_loss_disc
