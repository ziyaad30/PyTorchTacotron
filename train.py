"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --checkpoint=<name>  Restore model from checkpoint path if given.
    -h, --help                Show this help message and exit
"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import phoneme_to_sequence, symbols
from util import audio

# The tacotron model
from tacotron_pytorch import Tacotron

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser

import sys
import os
from hparams import hparams
from tqdm import tqdm
from util.logger import Tacotron2Logger

from synthesis import tts

# Default DATA_ROOT
DATA_ROOT = join(hparams.dataset_root, "training")

fs = hparams.sample_rate

global_step = 0
global_epoch = 0

use_cuda = torch.cuda.is_available()

if use_cuda:
    cudnn.benchmark = False
    device = 'cuda'
else:
    device = 'cpu'


def load_model(checkpoint_path):
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

def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


class TextDataSource(FileDataSource):
    def __init__(self):
        self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[-1], lines))
        return lines

    def collect_features(self, text):
        return np.asarray(phoneme_to_sequence(text),
                          dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(DATA_ROOT, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(MelSpecDataSource, self).__init__(1)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(LinearSpecDataSource, self).__init__(0)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int32)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, y_batch


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr


def train(model, data_loader, optimizer,
          init_lr=0.002,
          clip_thresh=1.0):
    
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()

    global global_step, global_epoch
    pbar = tqdm(total=hparams.run_steps, desc="Training", position=0)
    pbar.n = global_step
    
    while global_step < hparams.run_steps:
        running_loss = 0.
        for step, (x, input_lengths, mel, y) in enumerate(data_loader):
            # Decay learning rate
            current_lr = _learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            x, mel, y = x[indices], mel[indices], y[indices]

            # Feed data
            x, mel, y = Variable(x), Variable(mel), Variable(y)
            if use_cuda:
                x, mel, y = x.cuda(), mel.cuda(), y.cuda()
            mel_outputs, linear_outputs, attn = model(
                x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            if global_step > 0 and global_step % hparams.checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, hparams.checkpoint_dir, global_epoch)

            # Eval
            if global_step > 0 and global_step % hparams.eval_interval == 0:
                model.eval()
                waveform, alignment, _ = tts(model, hparams.sample_text)
                logger.plot_alignment(alignment.T, waveform, global_step)
                logger.sample_train(attn, sorted_lengths, linear_outputs, global_step)
                model.train()
            
            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
            optimizer.step()
            
            # Logging
            if global_step > 0 and global_step % hparams.log_interval == 0:
                logger.log_training(float(mel_loss.data), float(loss.data), grad_norm, current_lr, global_step)

            global_step += 1
            pbar.update(1)
            running_loss += loss.data

        averaged_loss = running_loss / (len(data_loader))
        logger.plot_losses(averaged_loss, global_epoch)
        pbar.write("Loss: {}".format(running_loss / (len(data_loader))))
        global_epoch += 1
        


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step_{}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    remove_checkpoints(checkpoint_dir)
    print("Saved checkpoint:", checkpoint_path)


def remove_checkpoints(checkpoint_dir):
    for filename in sorted(os.listdir(checkpoint_dir))[:-2]:
        filename_relPath = os.path.join(checkpoint_dir, filename)
        try:
            os.remove(filename_relPath)
        except OSError:
            pass


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["--checkpoint"]

    os.makedirs(hparams.checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    X = FileSourceDataset(TextDataSource())
    Mel = FileSourceDataset(MelSpecDataSource())
    Y = FileSourceDataset(LinearSpecDataSource())

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=len(symbols),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(os.path.join(hparams.checkpoint_dir, checkpoint_path))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"] + 1
            global_epoch = checkpoint["global_epoch"]
        except:
            # TODO
            pass

    # Setup tensorboard logger
    logger = Tacotron2Logger("log/run-test")

    # Train!
    try:
        train(model, data_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, hparams.checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
