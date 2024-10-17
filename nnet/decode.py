import argparse
import os
import pprint

import librosa
import numpy as np
import toml
import torch
from tqdm import tqdm

from libs.audio import WaveReader, write_wav
from libs.utils import get_logger
from pTFGridNet import pTFGridNet

logger = get_logger(__name__)


def load_obj(obj, device):
    def cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def get_comSTFT(egs, signal_configs, device):
    win_size = int(signal_configs['sr'] * signal_configs['win_size'])
    win_shift = int(signal_configs['sr'] * signal_configs['win_shift'])

    batch_mix_stft = torch.stft(egs['mix'],
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                window=torch.hann_window(win_size).to(device))  # (B,F,T,2)
    batch_mix_stft = torch.view_as_real(batch_mix_stft)
    batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1) ** signal_configs['beta'], \
        torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[..., 0])
    egs['mix'] = torch.stack((batch_mix_mag * torch.cos(batch_mix_phase),
                              batch_mix_mag * torch.sin(batch_mix_phase)), dim=-1).permute(0, 3, 2, 1)

    batch_aux_stft = torch.stft(egs['aux'],
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                window=torch.hann_window(win_size).to(device))  # (B,F,T,2)
    batch_aux_stft = torch.view_as_real(batch_aux_stft)
    batch_aux_mag, batch_aux_phase = torch.norm(batch_aux_stft, dim=-1) ** signal_configs['beta'], \
        torch.atan2(batch_aux_stft[..., -1], batch_aux_stft[..., 0])
    egs['aux'] = torch.stack((batch_aux_mag * torch.cos(batch_aux_phase),
                              batch_aux_mag * torch.sin(batch_aux_phase)), dim=-1).permute(0, 3, 2, 1)

    egs["aux_len"] = torch.tensor([(length - win_size + win_size) // win_shift + 1 for length in egs["aux_len"]],
                                  dtype=torch.int, device=device)  # center case

    return egs


class Decoder(object):
    def __init__(self, nnet, gpuid=(0, ), cpt_dir=None, configs=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)

        self.gpuid = gpuid
        self.configs_signal = configs['signal']
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.checkpoint = cpt_dir
        self.num_params = sum([param.nelement() for param in nnet.parameters()]) / 10.0 ** 6

        nnet = self._load_nnet(nnet, cpt_dir)
        self.nnet = nnet.to(self.device)
        self.nnet.eval()

        logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(gpuid, self.num_params))

    def _load_nnet(self, nnet, cpt_dir):
        cpt_fname = os.path.join(cpt_dir, 'checkpoint', "best.pt.tar")
        cpt = torch.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(cpt_fname, cpt["epoch"]))

        return nnet

    def estimate(self, egs):
        win_size = int(self.configs_signal['sr'] * self.configs_signal['win_size'])
        win_shift = int(self.configs_signal['sr'] * self.configs_signal['win_shift'])

        with torch.no_grad():
            egs = load_obj(egs, self.device)
            egs = get_comSTFT(egs, self.configs_signal, self.device)

            batch_est_stft, _ = self.nnet(egs["mix"], egs["aux"], egs["aux_len"])

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
                                        length=max(egs['valid_len'])).cpu().numpy()  # (B,L)

        return batch_est_wav


def run(data_type, cpt_dir, config):
    gpuids = tuple(config['gpu']['gpu_ids'])

    net = pTFGridNet(n_fft=configs['signal']['fft_num'],
                     n_layers=configs['net']['n_layers'],
                     lstm_hidden_units=configs['net']['lstm_hidden_units'],
                     attn_n_head=configs['net']['attn_n_head'],
                     attn_approx_qk_dim=configs['net']['attn_approx_qk_dim'],
                     emb_dim=configs['net']['emb_dim'],
                     emb_ks=configs['net']['emb_ks'],
                     emb_hs=configs['net']['emb_hs'],
                     num_spks=configs['path']['num_spks'],
                     activation=configs['net']['activation'],
                     eps=configs['net']['eps'])

    mix_input = WaveReader(os.path.join('data', data_type, 'real/mix.scp'), sample_rate=8000)
    aux_input = WaveReader(os.path.join('data', data_type, 'real/aux.scp'), sample_rate=8000)

    decoder = Decoder(net, gpuid=gpuids, cpt_dir=cpt_dir, configs=config)

    for key, mix in tqdm(mix_input, desc='Processing', dynamic_ncols=True, unit_scale=True):
        aux = aux_input[key]

        mix_std_, aux_std_ = np.std(mix), np.std(aux)
        mix, aux = mix / mix_std_, aux / aux_std_

        egs = {'mix': torch.tensor(mix, dtype=torch.float32).unsqueeze(0),
               'aux': torch.tensor(aux, dtype=torch.float32).unsqueeze(0),
               'aux_len': torch.tensor([len(aux)], dtype=torch.int),
               'valid_len': torch.tensor([len(mix)], dtype=torch.int)}

        est = decoder.estimate(egs)
        est = np.squeeze(est, axis=0) * mix_std_

        assert len(est) == len(mix), "Est length must be equal to Mix length!"
        write_wav(os.path.join(os.path.join('rec', data_type), "{}.wav".format(key)),
                  est / np.max(np.abs(est)),
                  fs=configs['signal']['sr'])

    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command to start pTFGridNet.py training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default="configs/train_config.toml",
                        help="Path to configs")
    parser.add_argument("--data_type",
                        type=str,
                        required=False,
                        default='wsj0_2mix_extr',
                        help="Name to datasets")
    parser.add_argument("--cpt_dir",
                        type=str,
                        required=False,
                        default='exp/pTFGridNet',
                        help="Name to datasets")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    configs = toml.load(args.config)
    print(configs)

    run(args.data_type, args.cpt_dir, configs)
