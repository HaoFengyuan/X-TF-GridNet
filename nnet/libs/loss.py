import torch


class SISDRLoss(object):
    def __init__(self,
                 scale_label=True,
                 zero_mean=True):
        self.zero_mean = zero_mean
        self.scale_label = scale_label

    def __call__(self, esti, label, length_list):
        def l2norm(x, keepdim=False):
            return torch.norm(x, dim=1, keepdim=keepdim)

        utt_num, chunk_length = esti.shape
        eps = 1e-5

        with torch.no_grad():
            mask_for_loss = torch.zeros((utt_num, chunk_length), dtype=label.dtype, device=label.device)
            for i in range(utt_num):
                mask_for_loss[i, :length_list[i]] = 1

        # Mask
        label = label[:, :chunk_length] * mask_for_loss
        esti = esti * mask_for_loss

        # Mean to zero
        if self.zero_mean:
            label = label - torch.mean(label, dim=1, keepdim=True)
            esti = esti - torch.mean(esti, dim=1, keepdim=True)

        # Scale
        if self.scale_label:
            scale_label = torch.sum(esti * label, dim=1, keepdim=True) * label / (
                        l2norm(label, keepdim=True) ** 2 + eps)
            loss = 20 * torch.log10(l2norm(scale_label) / (l2norm(esti - scale_label) + eps) + eps)
        else:
            scale_esti = torch.sum(esti * label, dim=1, keepdim=True) * esti / (
                    l2norm(esti, keepdim=True) ** 2 + eps)
            loss = 20 * torch.log10(l2norm(label) / (l2norm(scale_esti - label) + eps) + eps)

        return -torch.sum(loss) / utt_num
