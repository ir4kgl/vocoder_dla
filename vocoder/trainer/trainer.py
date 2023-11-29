import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from vocoder.base import BaseTrainer
from vocoder.utils import MetricTracker
from vocoder.mel.mel import MelSpectrogram, MelSpectrogramConfig

EVAL_DATA = ["./eval_data/mels/mel1.pt",
             "./eval_data/mels/mel2.pt",
             "./eval_data/mels/mel3.pt"]

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            discriminator,
            adv_criterion_g,
            adv_criterion_d,
            mel_criterion,
            fm_criterion,
            optimizer_g,
            optimizer_d,
            config,
            device,
            dataloaders,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            skip_oom=True,
            mel_config=None
    ):
        super().__init__(generator, discriminator,
                         adv_criterion_g, adv_criterion_d, mel_criterion, fm_criterion,
                         optimizer_g, optimizer_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]

        self.len_epoch = len(self.train_dataloader)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "mpd_loss", "msd_loss", "discriminator_loss",
            "Mel_loss", "FM_loss", "Adv_loss", "generator_loss",
            "grad norm", writer=self.writer
        )

        if mel_config == None:
            mel_config = MelSpectrogramConfig()
        self.mel_spec = MelSpectrogram(mel_config).to(device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.discriminator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.discriminator.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.discriminator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory

                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )

                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        with torch.no_grad():
            self._evaluation_epoch()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        batch["mel"] = self.mel_spec(batch["audio"]).squeeze()

        batch["audio_pred"] = self.generator(batch["mel"])
        batch["mel_pred"] = self.mel_spec(batch["audio_pred"])

        print(batch["audio"].shape)
        print(batch["audio_pred"].shape)
        print(batch["mel"].shape)
        print(batch["mel_pred"].shape)


        if is_train:
            self.optimizer_d.zero_grad()

        mpd_out, _, msd_out, _ = self.discriminator(batch["audio"])
        mpd_out_pred, _, msd_out_pred, _ = self.discriminator(batch["audio_pred"])

        batch["mpd_loss"] = self.adv_criterion_d(mpd_out, mpd_out_pred)
        batch["msd_loss"] = self.adv_criterion_d(msd_out, msd_out_pred)
        batch["discriminator_loss"] = batch["mpd_loss"] + batch["msd_loss"]
        if is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

        metrics.update("mpd_loss", batch["mpd_loss"].item())
        metrics.update("msd_loss", batch["msd_loss"].item())
        metrics.update("discriminator_loss", batch["discriminator_loss"].item())

        if is_train:
            self.optimizer_g.zero_grad()

        mpd_out, mpd_fm, msd_out, msd_fm = self.discriminator(batch["audio"])
        mpd_out_pred, mpd_fm_pred, msd_out_pred, msd_fm_pred = self.discriminator(batch["audio_pred"])

        batch["Adv_loss"] = self.adv_criterion_g(mpd_out_pred) + self.adv_criterion_g(msd_out_pred)
        batch["FM_loss"] = self.fm_criterion(mpd_fm, mpd_fm_pred) + self.fm_criterion(msd_fm, msd_fm_pred)
        batch["Mel_loss"] = self.mel_criterion(batch["mel"], batch["mel_pred"])

        batch["generator_loss"] = batch["FM_loss"] + batch["Mel_loss"] + batch["Adv_loss"]

        if is_train:
            batch["generator_loss"].backward()
            self._clip_grad_norm()
            self.optimizer_g.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

        metrics.update("Mel_loss", batch["Mel_loss"].item())
        metrics.update("FM_loss", batch["FM_loss"].item())
        metrics.update("Adv_loss", batch["Adv_loss"].item())
        metrics.update("generator_loss", batch["generator_loss"].item())

        return batch

    def _evaluation_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.discriminator.eval()
    
        for i, mel_path in enumerate(EVAL_DATA):
                mel = torch.load(mel_path)
                assert(mel.shape[1]) == 80
                audio = self.generator(mel).squeeze()
                self.writer.add_audio("synthesised_audio_{}".format(i), audio, sample_rate=22050)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters_g = self.generator.parameters()
        parameters_d = self.discriminator.parameters()
        if isinstance(parameters_g, torch.Tensor):
            parameters_g = [parameters_g]
        if isinstance(parameters_d, torch.Tensor):
            parameters_d = [parameters_d]
        parameters = parameters_g + parameters_d
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
