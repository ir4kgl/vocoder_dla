from torch.nn.functional import l1_loss

class MelLoss():
    def __call__(self, mel_gt, mel_preds):
        return l1_loss(mel_gt, mel_preds)
