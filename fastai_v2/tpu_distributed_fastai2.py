import warnings
warnings.filterwarnings('ignore')

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch

import fastai2
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.basic_train import *

def len_parallelloader(self): return len(self._loader._loader)
pl.PerDeviceLoader.__len__ = len_parallelloader
  

class TPUDistributed(Callback):
    def __init__(self, learn:Learner):
        self.device = xm.xla_device()
    
    def _change_dl(self,dl, shuffle):
        old_dl = dl
        sampler = torch.utils.data.distributed.DistributedSampler(
            dl.dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=shuffle)
        new_dl = dl.new(shuffle=False, sampler=sampler)
        return old_dl,new_dl,sampler


  def begin_fit(self):
        self.learn.model = self.learn.model.to(self.device)
        self.learn.opt.lr = self.learn.opt.lr*xm.xrt_world_size()

        shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(self.data.train_dl, 'init_kwargs') else True
        self.old_sampler_train_dl,self.data.train_dl,self.train_sampler = self._change_dl(self.data.train_dl, shuffle)
        if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
            self.old_sampler_valid_dl,self.data.valid_dl,self.valid_sampler = self._change_dl(self.data.valid_dl, shuffle)
  def begin_epoch(self,**kwargs:Any)->None:
    self.old_train_dl = self.data.train_dl
    self.learn.data.train_dl = pl.ParallelLoader(self.old_train_dl, [self.device]).per_device_loader(self.device)
    self.learn.data.train_dl.dataset = None #self.old_train_dl.dataset
    if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
      self.old_valid_dl = self.learn.data.valid_dl
      self.learn.data.valid_dl = pl.ParallelLoader(self.old_valid_dl, [self.device]).per_device_loader(self.device)      

      self.learn.data.valid_dl.dataset = self.old_valid_dl.dataset
      self.learn.data.valid_dl.dl = self.learn.data.valid_dl._loader._loader

    def after_backward(self):
        xm.optimizer_step(self.learn.opt)
        self.learn.opt.zero_grad()
        return CancelBatchException
    
    def after_epoch(self):
        self.learn.data.train_dl = self.old_train_dl
        self.learn.data.valid_dl = self.old_valid_dl
  
    def after_train(self):
        self.learn.data.train_dl = self.old_sampler_train_dl
        self.learn.data.valid_dl = self.old_sampler_valid_dl


def _to_tpu_distributed(learn:Learner) -> Learner:
    learn.callback_fns.append(TPUDistributed)
    return learn
  

Learner.to_tpu_distributed = _to_tpu_distributed
  

def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

path = untar_data(URLs.FOOD)
train_path = path/'train.txt'
test_path = path/'test.txt'

def train_loop():
    train_df = filelist2df(train_path)
    test_df = filelist2df(test_path)
    food = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_x = ColReader(1,pref=path/'images'),
                 splitter = RandomSplitter(),
                 get_y = ColReader(cols=0),
                 item_tfms=Resize(224),
                 batch_tfms=aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
                 )
    dls = food.dataloaders(train_df.values,bs=64)
    learn = cnn_learner(dls, models.resnet152, metrics=accuracy)
    learn.fit_tpu(3)

if __name__ == "__main__":
  xmp.spawn(train_loop,args=())

