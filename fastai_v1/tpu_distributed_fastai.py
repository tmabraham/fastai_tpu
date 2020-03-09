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

def len_parallelloader(self):
  return len(self._loader._loader)
pl.PerDeviceLoader.__len__ = len_parallelloader
  

class TPUDistributed(LearnerCallback):
  def __init__(self, learn:Learner):
    super().__init__(learn)
    self.device = xm.xla_device()

  def _change_dl(self,dl, shuffle):
    old_dl = dl
    sampler = torch.utils.data.distributed.DistributedSampler(
      dl.dataset,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=shuffle
    )
    new_dl = dl.new(shuffle=False, sampler=sampler)
    return old_dl,new_dl,sampler


  def on_train_begin(self, **kwargs:Any)->None:
    self.learn.model = self.learn.model.to(self.device)
    self.learn.opt.lr = self.learn.opt.lr*xm.xrt_world_size()

    shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(self.data.train_dl, 'init_kwargs') else True
    self.old_sampler_train_dl,self.data.train_dl,self.train_sampler = self._change_dl(self.data.train_dl, shuffle)
    if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
      self.old_sampler_valid_dl,self.data.valid_dl,self.valid_sampler = self._change_dl(self.data.valid_dl, shuffle)
  def on_epoch_begin(self,**kwargs:Any)->None:
    self.old_train_dl = self.data.train_dl
    self.learn.data.train_dl = pl.ParallelLoader(self.old_train_dl, [self.device]).per_device_loader(self.device)
    self.learn.data.train_dl.dataset = None #self.old_train_dl.dataset
    if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
      self.old_valid_dl = self.learn.data.valid_dl
      self.learn.data.valid_dl = pl.ParallelLoader(self.old_valid_dl, [self.device]).per_device_loader(self.device)      

      self.learn.data.valid_dl.dataset = self.old_valid_dl.dataset
      self.learn.data.valid_dl.dl = self.learn.data.valid_dl._loader._loader

  def on_backward_end(self, **kwargs:Any)->None:
    xm.optimizer_step(self.learn.opt)
    return {'skip_step': True}

  def on_epoch_end(self,**kwargs:Any)->None:
    self.learn.data.train_dl = self.old_train_dl
    self.learn.data.valid_dl = self.old_valid_dl
  
  def on_train_end(self,**kwargs:Any)->None:
    self.learn.data.train_dl = self.old_sampler_train_dl
    self.learn.data.valid_dl = self.old_sampler_valid_dl


def _to_tpu_distributed(learn:Learner) -> Learner:
  #Learner.fit = _fit_tpu
  learn.callback_fns.append(TPUDistributed)
  return learn
  

Learner.to_tpu_distributed = _to_tpu_distributed
  

path = untar_data(URLs.FOOD)
def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

train_path = path/'train.txt'
test_path = path/'test.txt'

def train_loop(index):
  train_df = filelist2df(train_path)
  test_df = filelist2df(test_path)


  data = (ImageList.from_df(df=train_df, path=path/'images', cols=1)
          .random_split_by_pct(0.2)
          .label_from_df(cols=0)
          .transform(get_transforms(),size=224)
          .databunch(bs=256, num_workers=4)
          .normalize(imagenet_stats))
  learn = cnn_learner(data, models.resnet152, metrics=accuracy).to_tpu_distributed()
  print('hello')
  learn.fit(4)

if __name__ == "__main__":
  xmp.spawn(train_loop,args=())

