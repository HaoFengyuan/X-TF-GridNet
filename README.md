# X-TF-GridNet: A Time-Frequency Domain Target Speaker Extraction Network with Adaptive Speaker Embedding Fusion

This project relates to the implementation of X-TF-GridNet, a Target Speaker Extraction Network (TSE) in the time-frequency (T-F) domain, which has been accepted by *Information Fusion*. Our proposed method boasts two key extensions: a U<sup>2</sup>-Net style network adeptly extracts robust fixed speaker embeddings, and an adaptive embedding fusion (AEA) mechanism ensures the effective utilization of target speaker information.

In this project, the primary basis is the original implementation of [SpEx+](https://github.com/gemengtju/SpEx_Plus) and the implementation of [EEND-Pytorch](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py). Notably, the project only encompasses the traing and inference phase. For specifics on data preparation, please refer to [there](https://github.com/xuchenglin28/speaker_extraction_SpEx). 

## Results

We choose the PESQ, SDR and SI-SDR results on the WSJ0-2mix dataset for further comparison with other time domain TSE method.

| Method | Year | Domain | Causal | Param. (M) | MACs (G/s) | PESQ $\uparrow$ | SDR (dB) $\uparrow$| SI-SDR (dB) $\uparrow$ |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Mixture | - | - | - | - | - | 2.02 | 0.2 | 0.0 |
| SpEx | 2020 | T | &#10008; | 10.79 | 3.55 | - | 16.3 | 15.8 |
| SpEx+ | 2020 | T | &#10008; | 11.14 | 3.76 | 3.43 | 17.2 | 16.9 |
| X-DPRNN | 2020 | T | &#10008; | 6.32 | 63.92 | - | - | 17.4 |
| SpEx++ | 2021 | T | &#10008; | 34.08 | 11.88 | 3.53 | 18.4 | 18.0 |
| SpEx<sub>pc</sub> | 2021 | T | &#10008; | 28.40 | 40.54 | - | 18.8 | 18.6 |
| VEVEN | 2023 | T | &#10008; | 2.63 | 85.11 | 3.66 | 19.2 | 19.0 |
| X-SepFormer | 2023 | T | &#10008; | 26.66 | 61.34 | <u>3.74</u> | 19.5 | 18.9 |
| X-TF-GridNet | 2023 | T-F | &#10008; | 7.79 | 68.32 | 3.70 | <u>20.4</u> | <u>19.7</u> |
| X-TF-GridNet (Large) | 2023 | T-F | &#10008; | 12.68 | 113.24 | __3.77__ | __21.7__ | __20.7__ |

(\* More details can be found in the paper.)


## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```bibtex
@article{hao2024if,
    title = {{X-TF-GridNet}: A time–frequency domain target speaker extraction network with adaptive speaker embedding fusion},
    journal = {Information Fusion},
    volume = {112},
    pages = {102550},
    year = {2024},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2024.102550},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253524003282},
    author = {Fengyuan Hao and Xiaodong Li and Chengshi Zheng},
}
```
