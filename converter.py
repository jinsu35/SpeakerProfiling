import torch
import torch.utils.data as data
import pytorch_lightning as pl
import coremltools as ct

import os
from argparse import ArgumentParser

from config import TIMITConfig
from TIMIT.lightning_model import LightningModel
from TIMIT.dataset import TIMITDataset

parser = ArgumentParser(add_help=True)
parser.add_argument('--data_path', type=str, default=TIMITConfig.data_path)
parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
parser.add_argument('--timit_wav_len', type=int, default=TIMITConfig.timit_wav_len)
parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
parser.add_argument('--alpha', type=float, default=TIMITConfig.alpha)
parser.add_argument('--beta', type=float, default=TIMITConfig.beta)
parser.add_argument('--gamma', type=float, default=TIMITConfig.gamma)
parser.add_argument('--hidden_size', type=float, default=TIMITConfig.hidden_size)
parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
parser.add_argument('--dev', type=str, default=False)
parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
parser.add_argument('--noise_dataset_path', type=str, default=TIMITConfig.noise_dataset_path)
parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
parser.add_argument('--training_type', type=str, default=TIMITConfig.training_type)
parser.add_argument('--data_type', type=str, default=TIMITConfig.data_type)

parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

model = LightningModel(vars(hparams))
model.load_from_checkpoint("speaker_age.ckpt")
model.eval()
train_set = TIMITDataset(
    wav_folder = os.path.join(hparams.data_path, 'TRAIN'),
    hparams = hparams
)
sample, _, _, _ = train_set[0]
print('sample shape', sample.shape)
model = torch.jit.trace(model, sample)

# Convert to Core ML using the Unified Conversion API
core_model = ct.convert(
    model,
    inputs=[ct.TensorType(shape=sample.shape)]
)

core_model.save("SpeakerAgeClassifier.mlmodel")
