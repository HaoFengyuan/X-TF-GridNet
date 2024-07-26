import difflib
import json
import logging
import os

import numpy as np
import torch


def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format="%Y-%m-%d %H:%M:%S",
               file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler() if not file else logging.FileHandler(name, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def dump_json(obj, fdir, name):
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)

    return obj


def sisnr(est, label):
    label_power = np.sum(label ** 2.0) + 1e-8
    scale = np.sum(est * label) / label_power

    est_true = scale * label
    est_res = est - est_true

    true_power = np.sum(est_true ** 2.0, axis=0) + 1e-8
    res_power = np.sum(est_res ** 2.0, axis=0) + 1e-8
    sdr = 10 * np.log10(true_power) - 10 * np.log10(res_power)

    return sdr


def cal_pesq(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id, :], esti_utts[id, :]

    from pypesq import pesq
    pesq_score = pesq(clean_utt, esti_utt, fs=fs)

    return pesq_score


def cal_stoi(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id, :], esti_utts[id, :]

    from pystoi import stoi
    stoi_score = stoi(clean_utt, esti_utt, fs, extended=True)

    return 100 * stoi_score


def cal_sisnr(id, esti_utts, clean_utts, real_length):
    clean_utt, esti_utt = clean_utts[id, :real_length[id]], esti_utts[id, :real_length[id]]
    sisnr_score = sisnr(esti_utt, clean_utt)

    return sisnr_score


def get_layer(l_name,
              library=torch.nn):
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError("Layer with name {} not found in {}.\n Closest matches: {}".format(
            l_name, str(library), close_matches))
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(l_name, [x.lower() for x in all_torch_layers])
        raise NotImplementedError("Multiple matchs for layer with name {} not found in {}.\n All matches: {}".format(
            l_name, str(library), close_matches))
    else:
        layer_handler = getattr(library, match[0])

        return layer_handler
