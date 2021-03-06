# Fre-GAN: Adversarial Frequency-consistent Audio Synthesis
### Ji-Hoon Kim, Sang-Hoon Lee, Ji-Hyun Lee, Seong-Whan Lee

In our recent [paper](https://arxiv.org/abs/2106.02297), we presented Fre-GAN: a GAN-based neural vocoder that is able to generated high-quality speech from mel-spectrogram.
We provide the official implementation of Fre-GAN and pretrained network parameters in this repository.

**Abstract:** Although recent works on neural vocoder have improved the quality of synthesized audio, there still exists a gap between generated and ground-truth audio in frequency space. This difference leads to spectral artifacts such as hissing noise or robotic sound, and thus degrades the sample quality. In this paper, we propose Fre-GAN which achieves frequency-consistent audio synthesis with highly improved generation quality. Specifically, we first present resolution-connected generator and resolution-wise discriminators, which help learn various scales of spectral distributions over multiple frequency bands. Additionally, to reproduce high-frequency components accurately, we leverage discrete wavelet transform in the discriminators. From our experiments, Fre-GAN achieves high-fidelity waveform generation with a gap of only 0.03 MOS compared to ground-truth audio while outperforming standard models in quality. 

Visit our [demo page](http://prml-lab-speech-team.github.io/demo/FreGAN) for audio samples.

## Requirements
<ol>
<li>Clone this respository
<pre>
<code>git clone https://github.com/lism13/Fre-GAN.git</code>
</pre>
<li>Install python requirememts.
<pre>
<code>pip3 install -r requirements.txt</code>
</pre>
  <li>Download and extract the <a href='https://keithito.com/LJ-Speech-Dataset/'>LJSpeech </a> dataset. And move all wave filese to LJSpeech-1.1/wavs
</ol>

## Training
<pre>
<code> python train.py --config config_v1.json</code>
</pre>

## Pre-tarined Model
You can simply use pretrained models we provide.
[Download pretrained model]('')

## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow) and [HiFi-GAN](https://github.com/jik876/hifi-gan) to implement this model. 
