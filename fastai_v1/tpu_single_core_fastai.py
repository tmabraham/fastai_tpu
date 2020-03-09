import warnings
warnings.filterwarnings('ignore')

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch

import fastai
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.basic_train import *

class TPUSingleCore(LearnerCallback):
  def __init__(self, learn:Learner):
    super().__init__(learn)
    self.device = xm.xla_device()

  def on_train_begin(self, **kwargs:Any)->None:
    self.learn.model = self.learn.model.to(self.device)

  def on_batch_begin(self, last_input, last_target, train, **kwargs):
    return {'last_input': last_input.to(self.device), 'last_target': last_target.to(self.device)}

  def on_backward_end(self, **kwargs:Any)->None:
    xm.optimizer_step(self.learn.opt)
    return {'skip_step': True}


def _to_tpu_single(learn:Learner) -> Learner:
  #Learner.fit = _fit_tpu
  learn.callback_fns.append(TPUSingleCore)
  return learn
  

Learner.to_tpu_distributed = _to_tpu_single
  

path = untar_data(URLs.FOOD)
def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

train_path = path/'train.txt'
test_path = path/'test.txt'

def train_loop():
  train_df = filelist2df(train_path)
  test_df = filelist2df(test_path)


  data = (ImageList.from_df(df=train_df, path=path/'images', cols=1)
          .random_split_by_pct(0.2)
          .label_from_df(cols=0)
          .transform(get_transforms(), size=224)
          .databunch(bs=512, num_workers=16)
          .normalize(imagenet_stats))
  learn = cnn_learner(data, models.resnet152, metrics=accuracy).to_tpu_distributed()
  print('hello')
  learn.fit(3)

train_loop()

