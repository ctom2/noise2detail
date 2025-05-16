# Noise2Detail: Lightweight Data-Free Denoising for Detail-Preserving Biomedical Image Restoration

#### [[`Paper`]()] [[`Colab demo`]()] [[`BibTeX`](#citing-noise2detail)]

<blockquote>
  <p align="left">
    <p align="left">
      <a href='https://chobola.ai/' target='_blank'>Tomáš Chobola</a>*&emsp;
      <a href='https://scholar.google.de/citations?user=FPykfZ0AAAAJ&hl=cs&oi=ao' target='_blank'>Julia A. Schnabel</a>&emsp;
      <a href='https://scholar.google.de/citations?user=jUiKc6QAAAAJ&hl=cs&oi=sra' target='_blank'>Tingying Peng</a>*&emsp;
      <br>
      Technical University of Munich&emsp;Helmholtz AI&emsp;King’s College London
    </p>
  </p>
</blockquote>

\* Corresponding author

Accepted early to **MICCAI 2025**.

## Overview

- **Ultra-Lightweight Model Design:** Introducing a computationally efficient model that achieves fast denoising with minimal memory demands, balancing inference speed and high-quality image restoration.
- **Noise2Detail (N2D) Pipeline:** Proposes an innovative multistage denoising approach within the Noise2Noise framework, disrupting noise correlations to generate intermediate smooth structures and refining them to recover fine details from noisy inputs.
- **Dataset-Free Denoising:** Eliminates the need for clean reference images or explicit noise modeling, making it ideal for biomedical imaging where clean training data is scarce due to rare and complex modalities.
- **Superior Performance with Low Resources:** Outperforms existing dataset-free denoising techniques while requiring significantly fewer computational resources, enabling practical use in real-world applications.

## Citing Noise2Detail

Please consider citing our paper if our code are useful:

```bibtex
@inproceedings{Chobola2025,
  title = {Lightweight Data-Free Denoising for Detail-Preserving Biomedical Image Restoration},
  author = {Chobola,  Tomáš and Schnabel,  Julia A. and Peng,  Tingying},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025}
}

```
