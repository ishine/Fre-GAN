# Fre-GAN: Adversarial Frequency-consistent Audio Synthesis
### Ji-Hoon Kim, Sang-Hoon Lee, Ji-Hyun Lee, Seoung-Whan Lee

In our recent paper, we propose Fre-GAN: a GAN-based neural vocoder that is able to generated high-quality speech from mel-spectrogram.
We provide the official implementation of Fre-GAN and pretrained network parameters in this repository.

**Abstract:** Although recent works on neural vocoder have improved the quality of synthesized audio, there still exists a gap between generated and ground-truth audio in frequency space. This difference leads to spectral artifacts such as hissing noise or reverberation, and thus degrades the sample quality. In this paper, we propose Fre-GAN which achieves frequency-consistent audio synthesis with highly improved generation quality. Specifically, we first present resolution-connected generator and resolution-wise discriminators, which help learn various scales of spectral distributions over multiple frequency bands. Additionally, to reproduce high-frequency components accurately, we leverage discrete wavelet transform in the discriminators. From our experiments, Fre-GAN achieves high-fidelity waveform generation with a gap of only 0.03 MOS compared to ground-truth audio while outperforming standard models in quality. 

Visit our [demo page](http://prml-lab-speech-team.github.io/demo/FreGAN) for audio samples.

## Requirements
1. Clone this respository
> git clone
3. Install python requirememts.
