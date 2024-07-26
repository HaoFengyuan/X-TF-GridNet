from itertools import permutations

import numpy as np


def si_snr(label, esti, remove_dc=True, scale_label=True):
    if scale_label:
        return si_snr_label(label, esti, remove_dc=remove_dc)
    else:
        return si_snr_esti(label, esti, remove_dc=remove_dc)


def si_snr_label(label, esti, remove_dc=True):
    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # Mean to zero
    if remove_dc:
        label = label - np.mean(label)
        esti = esti - np.mean(esti)

    # Scale
    scale_label = np.inner(esti, label) * label / (vec_l2norm(label) ** 2)

    return 20 * np.log10(vec_l2norm(scale_label) / vec_l2norm(scale_label - esti))


def si_snr_esti(label, esti, remove_dc=True):
    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # Mean to zero
    if remove_dc:
        label = label - np.mean(label)
        esti = esti - np.mean(esti)

    # Scale
    scale_esti = np.inner(esti, label) * esti / (vec_l2norm(esti) ** 2)

    return 20 * np.log10(vec_l2norm(label) / vec_l2norm(label - scale_esti))


def permute_si_snr(xlist, slist):
    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(N, len(slist)))

    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))

    return max(si_snrs)
