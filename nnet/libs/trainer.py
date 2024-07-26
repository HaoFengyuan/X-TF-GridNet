import os
import time

import hdf5storage
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .audio import get_comSTFT
from .loss import SISDRLoss
from .utils import get_logger, cal_sisnr


def load_obj(obj, device):
    def cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.unpro_metric = []
        self.pro_metric = []
        self.timer = SimpleTimer()

    def add(self, loss, metric_dict=None):
        self.loss.append(loss)
        if metric_dict is not None:
            self.unpro_metric.append(metric_dict['unpro_metric'])
            self.pro_metric.append(metric_dict['pro_metric'])

        N = len(self.loss)
        if not N % self.period:
            avg_loss = sum(self.loss[-self.period:]) / self.period
            if metric_dict is not None:
                avg_unpro_metric = sum(self.unpro_metric[-self.period:]) / self.period
                avg_pro_metric = sum(self.pro_metric[-self.period:]) / self.period
                self.logger.info(
                    "Processed {:d} batches (loss = {:+.4f}, unpro_metric = {:+.2f}, pro_metric = {:+.2f})..."
                    .format(N, avg_loss, avg_unpro_metric, avg_pro_metric))
            else:
                self.logger.info("Processed {:d} batches (loss = {:+.4f})...".format(N, avg_loss))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))

            if self.pro_metric:
                metric_impr = [pro_metric - unpro_metric
                               for pro_metric, unpro_metric in zip(self.pro_metric, self.unpro_metric)]
                sstr = ",".join(map(lambda f: "{:.2f}".format(f), metric_impr))
                self.logger.info("Loss on {:d} batches: {}".format(N, sstr))

        if self.pro_metric:
            return {"loss": sum(self.loss) / N,
                    "metric_impr": sum(metric_impr) / N,
                    "batches": N,
                    "cost": self.timer.elapsed()}
        else:
            return {"loss": sum(self.loss) / N,
                    "batches": N,
                    "cost": self.timer.elapsed()}


class Trainer(object):
    def __init__(self,
                 net,
                 gpuid=0,
                 configs=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)

        self.gpuid = gpuid
        self.configs_signal = configs['signal']
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.checkpoint = configs['save']['save_filename']
        self.clip_norm = configs['optimizer']['gradient_norm']
        self.logging_period = configs['optimizer']['logging_period']
        self.no_impr = configs['optimizer']['early_stop_freq']
        self.resume = configs['path']['resume_filename']
        self.optimizer = configs['optimizer']['name']
        self.rel_epsilon = 1. - configs['optimizer']['rel_epsilon']
        self.num_params = sum([param.nelement() for param in net.parameters()]) / 10.0 ** 6
        self.cur_epoch = 0

        self.SISDRloss = SISDRLoss(zero_mean=configs['loss_function']['zero_mean'],
                                   scale_label=configs['loss_function']['scale_label'])

        if self.checkpoint and not os.path.exists(self.checkpoint):
            os.makedirs(os.path.join(self.checkpoint, 'checkpoint'))
        self.logger = get_logger(
            os.path.join(self.checkpoint, "trainer.log"), file=True)

        if self.resume:
            if not os.path.exists(self.resume):
                raise FileNotFoundError("Could not find resume checkpoint: {}".format(self.resume))
            cpt = torch.load(self.resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(self.resume, self.cur_epoch))

            net.load_state_dict(cpt["model_state_dict"])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(self.optimizer,
                                                   configs['optimizer'],
                                                   state=cpt["optim_state_dict"])
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(self.optimizer,
                                                   configs['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode="min",
                                           factor=configs['optimizer']['factor'],
                                           patience=configs['optimizer']['halve_freq'],
                                           min_lr=configs['optimizer']['min_lr'],
                                           verbose=True)

        self.logger.info("Model summary:\n{}".format(net))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(gpuid, self.num_params))
        if self.clip_norm:
            self.logger.info("Gradient clipping by {}, default L2".format(self.clip_norm))

    def save_checkpoint(self, name, best=True):
        cpt = {"epoch": self.cur_epoch,
               "model_state_dict": self.net.state_dict(),
               "optim_state_dict": self.optimizer.state_dict()}
        
        torch.save(cpt, os.path.join(self.checkpoint, 'checkpoint', "{0}.pt.tar"
                                     .format("best" if best else 'epoch_' + name)))

    def create_optimizer(self, optimizer, config, state=None):
        supported_optimizer = {"adam": torch.optim.Adam,
                               "adamw": torch.optim.AdamW, }

        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        
        opt = supported_optimizer[optimizer](self.net.parameters(),
                                             lr=config["lr"],
                                             betas=(config["beta1"], config["beta2"]),
                                             weight_decay=config["l2"])
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, config))

        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
            
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.net.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            egs = load_obj(egs, self.device)
            egs = get_comSTFT(egs, self.configs_signal, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)[-1]
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)

            self.optimizer.step()

            reporter.add(loss.item())

        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.net.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with torch.no_grad():
            for egs in data_loader:
                batch_clean_wav = egs['ref'].numpy()
                batch_mix_wav = egs['mix'].numpy()

                egs = load_obj(egs, self.device)
                egs = get_comSTFT(egs, self.configs_signal, self.device)
                batch_est_wav, loss = self.compute_loss(egs)

                batch_est_wav = batch_est_wav.cpu().numpy()

                metric_dict = {}
                unpro_score_list, pro_score_list = [], []
                for id in range(len(egs['valid_len'])):
                    unpro_score_list.append(cal_sisnr(id, batch_mix_wav, batch_clean_wav, egs['valid_len']))
                    pro_score_list.append(cal_sisnr(id, batch_est_wav, batch_clean_wav, egs['valid_len']))
                unpro_score_list, pro_score_list = np.asarray(unpro_score_list), np.asarray(pro_score_list)
                unpro_sisnr_mean_score, pro_sisnr_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
                metric_dict["unpro_metric"] = unpro_sisnr_mean_score
                metric_dict["pro_metric"] = pro_sisnr_mean_score

                reporter.add(loss.item(), metric_dict)

        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        with torch.cuda.device(self.gpuid[0]):
            stats = dict()

            self.save_checkpoint(name=str(self.cur_epoch), best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}, METRIC_IMPR = {:.2f}"
                             .format(self.cur_epoch, best_loss, cv["metric_impr"]))
            no_impr = 0
            self.scheduler.best = best_loss

            train_epoch, val_epoch, metric_impr_epoch = [], [cv["loss"]], [cv["metric_impr"]]
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats["title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(cur_lr, self.cur_epoch)

                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(tr["loss"], tr["cost"], tr["batches"])
                train_epoch.append(tr["loss"])

                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(cv["loss"], cv["cost"], cv["batches"])
                stats["metric"] = "metric impr = {:+.2f}".format(cv["metric_impr"])
                val_epoch.append(cv["loss"])
                metric_impr_epoch.append(cv["metric_impr"])

                stats["scheduler"] = ""
                if cv["loss"] >= best_loss * self.rel_epsilon:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(name=str(self.cur_epoch), best=True)
                self.logger.info("{title} {tr} | {cv} | {metric} {scheduler}".format(**stats))

                self.scheduler.step(cv["loss"])
                self.save_checkpoint(name=str(self.cur_epoch), best=False)
                if no_impr == self.no_impr:
                    self.logger.info("Stop training cause no impr for {:d} epochs".format(no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(self.cur_epoch, num_epochs))
            hdf5storage.savemat(os.path.join(self.checkpoint, 'loss_mat'),
                                {'train': train_epoch, 'val': val_epoch, 'metric_impr': metric_impr_epoch})


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)
        self.CrossEntropy = torch.nn.CrossEntropyLoss()

    def SISDR_loss(self, esti, label, length_list):
        return self.SISDRloss(esti, label, length_list)

    def compute_loss(self, egs):
        batch_est_stft, spk_pred = self.net(egs["mix"], egs["aux"], egs["aux_len"])

        win_size = int(self.configs_signal['sr'] * self.configs_signal['win_size'])
        win_shift = int(self.configs_signal['sr'] * self.configs_signal['win_shift'])

        batch_est_stft = batch_est_stft.permute(0, 3, 2, 1)
        batch_spec_mag, batch_spec_phase = torch.norm(batch_est_stft, dim=-1) ** (1 / self.configs_signal['beta']), \
            torch.atan2(batch_est_stft[..., -1], batch_est_stft[..., 0])
        batch_est_stft = torch.stack((batch_spec_mag * torch.cos(batch_spec_phase),
                                      batch_spec_mag * torch.sin(batch_spec_phase)), dim=-1)
        batch_est_stft = torch.view_as_complex(batch_est_stft)
        batch_est_wav = torch.istft(batch_est_stft,
                                    n_fft=self.configs_signal['fft_num'],
                                    hop_length=win_shift,
                                    win_length=win_size,
                                    return_complex=False,
                                    window=torch.hann_window(win_size).to(self.device),
                                    length=max(egs['valid_len']))  # (B,L)
        batch_est_wav = batch_est_wav * egs['mix_scale']

        SISDR_loss = self.SISDR_loss(batch_est_wav, egs["ref"], egs['valid_len'])
        ce_loss = self.CrossEntropy(spk_pred, egs["spk_idx"])

        return batch_est_wav, SISDR_loss + 0.5 * ce_loss
