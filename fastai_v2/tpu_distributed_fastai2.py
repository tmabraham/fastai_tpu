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
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.distributed import *

#This is only needed for PyTorch XLA 1.5
#@patch
#def __len__(self: pl.PerDeviceLoader):
#    return len(self._loader._loader)

def _fa_rebuild_tensor    (cls, *args, **kwargs): return cls(torch._utils._rebuild_tensor_v2 (*args, **kwargs))
def _fa_rebuild_qtensor   (cls, *args, **kwargs): return cls(torch._utils._rebuild_qtensor   (*args, **kwargs))
def _fa_rebuild_xla_tensor(cls, *args, **kwargs): return cls(torch._utils._rebuild_xla_tensor(*args, **kwargs))

@patch
def __reduce_ex__(self:TensorBase, proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        if self.device.type == 'xla': 
            args = (type(self), self.cpu().numpy(), self.dtype, str(self.device), self.requires_grad)
            return (_fa_rebuild_xla_tensor, args)
        
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        f = _fa_rebuild_qtensor if self.is_quantized else  _fa_rebuild_tensor
        return (f, args + (self.requires_grad, OrderedDict()))

@patch
def __getstate__(self: Optimizer):
        optim_dict = self.__dict__.copy()
        modified_dict = {**optim_dict, 'param_groups': self.param_groups} #this change needed since PyTorch XLA wants it!
        return modified_dict

@patch
def __setstate__(self: Optimizer,state):
        print('setstate Optimizer dict: ', self.__dict__.keys())
        del state['param_groups']
        self.__dict__.update(state)

@patch
def set_epoch(self: pl.PerDeviceLoader,epoch): 
       self._loader._loader.set_epoch(epoch)



# Much of the below code is inspired by the GPU distributed callback
class TPUDistributed(Callback):
    def __init__(self):
        self.device = xm.xla_device()

    def _wrap_dl(self, dl):
        if isinstance(dl, pl.PerDeviceLoader):
            return dl
        else:
            #dl = dl.to(self.device)
            dl.fake_l.num_workers=0 # For some reason, needed for it to work (something on fastai's end). Need to investigate further
            distributed_dl = DistributedDL.from_dl(dl, xm.get_ordinal(), xm.xrt_world_size()) # Use existing distributed functionality 
            distributed_dl.epoch=0
            return pl.ParallelLoader(distributed_dl, [self.device]).per_device_loader(self.device)

    def begin_fit(self):
        xm.master_print('begin fit')
        self.learn.model = self.learn.model.to(self.device)
        for h in self.opt.hypers: h['lr'] *= xm.xrt_world_size()
        self.old_dls = list(self.dls)
        print('wrapping dls')
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
        for dl in self.dls: dl.set_epoch(self.epoch)

    def begin_epoch(self):
        for dl in self.dls: dl.set_epoch(self.epoch)

    def begin_train(self):    
        self.learn.dl = self._wrap_dl(self.learn.dl)

    def begin_batch(self):
       self.learn.xb = [xb_item.to(self.device) for xb_item in self.xb]
       self.learn.yb = [yb_item.to(self.device) for yb_item in self.yb]
    def after_backward(self):
        xm.optimizer_step(self.learn.opt)
        self.learn.opt.zero_grad()
        return CancelBatchException


    def begin_validate(self): self.learn.dl = self._wrap_dl(self.learn.dl)

    def after_fit(self):
        self.learn.dls.loaders = self.old_dls

@patch
def to_tpu_distributed(self:Learner):
    self.add_cbs([TPUDistributed()])
    return self



def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

path = untar_data(URLs.FOOD)
train_path = path/'train.txt'
test_path = path/'test.txt'


def train_loop(index):
    print('index: ',index)
    train_df = filelist2df(train_path)
    test_df = filelist2df(test_path)
    food = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_x = ColReader(1,pref=path/'images'),
                 splitter = RandomSplitter(),
                 get_y = ColReader(cols=0),
                 item_tfms=Resize(224)
#                 batch_tfms=aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.) <-- ignore batch (on-TPU) tfms for now
                 )
    dls = food.dataloaders(train_df.values,bs=16)
    learn = cnn_learner(dls, resnet152, metrics=accuracy).to_tpu_distributed()
    learn.fit(3)

if __name__ == "__main__":
  xmp.spawn(train_loop,nprocs=1,args=())
