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
from fastai.data.load import _FakeLoader

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

def _round_to_multiple(number,multiple): return int(math.ceil(number/multiple)*multiple)

class TPUDistributedDL(TfmdDL):
    "A `TfmdDL` which splits a batch into equal size pieces for each TPU core"
    def __init__(self,dl,rank,world_size):
        store_attr()
        self.bs,self.device,self.num_workers,self.drop_last,self.dataset,self.offs,fake = \
            attrgetter('bs','device','num_workers','drop_last','dataset','offs','fake_l')(dl)
        self.fake_l = _FakeLoader(self, fake.pin_memory, fake.num_workers, fake.timeout, persistent_workers=fake.persistent_workers)
        self.SERIAL_EXEC = xmp.MpSerialExecutor()

    def _to_detach(self,b,cpu=True,gather=True): return to_detach(b,cpu,gather) # member func so we can override for test
    def __len__(self): return _round_to_multiple(len(self.dl),self.world_size)//self.world_size
    def get_idxs(self):
        idxs = self.SERIAL_EXEC.run(self.dl.get_idxs) # compute get_idxs in all ranks (we'll only use rank 0 but size must be consistent)
        self.n = len(idxs)              # we assumed n was dl.n but we really care about number of idxs
        # add extra samples to make it evenly divisible
        self.n_padded = _round_to_multiple(self.n,self.world_size)
        idxs += (idxs * (self.n_padded//self.n))[:self.n_padded-self.n] # idx needs to be repeated when n_padded>>n
        # slice padded idxs so that each rank gets self.n_padded//self.world_size tensors
        return idxs[self.rank*self.n_padded//self.world_size:(self.rank+1)*self.n_padded//self.world_size]

    def before_iter(self):
        self.i = 0
        self.dl.before_iter()

    def randomize(self): self.dl.randomize()
    def after_batch(self,b):
        self.i += find_bs(b)
        return self.dl.after_batch(b)

    def after_iter(self):  self.dl.after_iter()
    def create_batches(self,samps): return self.dl.create_batches(samps)
    def to_detach(self,b, cpu=True, gather=True):
        b = self._to_detach(b, cpu, gather)
        def _inner(b):
            if b.ndim>0:
                # for each rank, compute overflow of read idxs vs self.n and accumulate them to unpad totals after gathering
                n = sum([min(0,max(-len(b)//self.world_size,
                                   self.n-(self.i+r*self.n_padded//self.world_size))) for r in range(self.world_size)])
                b = b[:n or None]
            return b
        return apply(_inner,b) if gather and all(hasattr(self,o) for o in ('i','n','n_padded')) else b



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
            distributed_dl = TPUDistributedDL(dl, xm.get_ordinal(), xm.xrt_world_size()) # Use existing distributed functionality
            return pl.ParallelLoader(distributed_dl, [self.device]).per_device_loader(self.device)

    def before_fit(self):
        xm.master_print('begin fit')
        self.learn.model = self.learn.model.to(self.device)
        for h in self.opt.hypers: h['lr'] *= xm.xrt_world_size()
        self.old_dls = list(self.dls)
        print('wrapping dls')
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
#        for dl in self.dls: dl.set_epoch(self.epoch)

    #def before_epoch(self):
    #    for dl in self.dls: dl.set_epoch(self.epoch)

    def before_train(self):
        self.learn.dl = self._wrap_dl(self.learn.dl)

    def before_batch(self):
       self.learn.xb = [xb_item.to(self.device) for xb_item in self.xb]
       self.learn.yb = [yb_item.to(self.device) for yb_item in self.yb]
    def after_backward(self):
        xm.optimizer_step(self.learn.opt)
        self.learn.opt.zero_grad()
        return CancelBatchException


    def before_validate(self): self.learn.dl = self._wrap_dl(self.learn.dl)

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
    dls = food.dataloaders(train_df.values,bs=256)
    learn = cnn_learner(dls, resnet152, metrics=accuracy).to_tpu_distributed()
    learn.fit(3)

if __name__ == "__main__":
  xmp.spawn(train_loop,nprocs=8,args=())
