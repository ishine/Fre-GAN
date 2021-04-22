from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset_custom import mel_spectrogram, MAX_WAV_VALUE, load_wav
from model_0217 import Generator
import numpy as np
import time

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict



def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    #filelist = os.listdir(a.input_wavs_dir)
    filelist = sorted(glob.glob(os.path.join(a.input_wavs_dir, '*.npy')), reverse=True)[:100]
    # filelist = [a.input_wavs_dir]
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):

            # x = torch.from_numpy(np.load(filname)['mel']).unsqueeze(0).transpose(-1, -2).cuda()
            x = torch.from_numpy(np.load(filname)).unsqueeze(0).transpose(-1, -2).cuda()

            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            audio = audio/np.abs(audio).max()

            output_file = os.path.join(a.output_dir, filname.split('/')[-1] + '.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='./ttt/KSS1') # npy
    parser.add_argument('--output_dir', default='./tttt')
    parser.add_argument('--checkpoint_file', default='//hdd0/JIHOON/hyundai_log/fre_hyundai_22050_unnorm_e5/g_01100000', required=False)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

