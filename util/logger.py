import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from hparams import hparams
from util import audio

def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)
    
def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1)


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs = 5)

    def log_training(self, mel_loss, loss, grad_norm, learning_rate, iteration):
        self.add_scalar('mel loss', mel_loss, iteration)
        self.add_scalar('linear loss', loss, iteration)
        self.add_scalar('gradient norm', grad_norm, iteration)
        self.add_scalar('learning rate', learning_rate, iteration)

    def plot_losses(self, loss, iteration):
        self.add_scalar('training loss', loss, iteration)
        
    def sample_train(self, attn, input_lengths, linear_outputs, iteration):
        idx = min(1, len(input_lengths) - 1)
        alignment = attn[idx].cpu().data.numpy()
        linear_output = linear_outputs[idx].cpu().data.numpy()
        
        # plot alignment, mel and postnet output
        self.add_image('train.align', plot_alignment_to_numpy(alignment), iteration)
        self.add_image( 'train.mel', plot_spectrogram_to_numpy(linear_output.T), iteration)
        
        
        # Predicted audio signal
        signal = audio.inv_spectrogram(linear_output.T)
        self.add_audio('train.wav', signal, iteration, hparams.sample_rate)
        
    def plot_alignment(self, alignment, wav, iteration):
        self.add_image('infer.align', plot_alignment_to_numpy(alignment), iteration)
        self.add_audio('infer.wav', wav, iteration, hparams.sample_rate)