# coding: utf-8
"""
Synthesis waveform from trained model.

usage: tts.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    -h, --help               Show help message.
"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols
from util import audio

import torch
from torch.autograd import Variable
import numpy as np
import nltk

from tacotron_pytorch import Tacotron
from hparams import hparams

from tqdm import tqdm

use_cuda = torch.cuda.is_available()


def tts(model, text):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()
    # TODO: Turning off dropout of decoder's prenet causes serious performance
    # regression, not sure why.
    # model.decoder.eval()
    model.encoder.eval()
    model.postnet.eval()

    sequence = np.array(text_to_sequence(text, [hparams.cleaners]))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if use_cuda:
        sequence = sequence.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments = model(sequence)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram


if __name__ == "__main__":
    checkpoint_path = 'checkpoints/checkpoint_step_16055.pth'
    dst_dir = 'results'
    max_decoder_steps = 1000
    file_name_suffix = '_'

    model = Tacotron(n_vocab=len(symbols),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)


    text = 'Massino had two brothers John and Anthony.'
    words = nltk.word_tokenize(text)
    print("{} ({} chars, {} words)".format(text, len(text), len(words)))
    waveform, alignment, _ = tts(model, text)
    dst_wav_path = join(dst_dir, "result.wav".format(file_name_suffix))
    audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
