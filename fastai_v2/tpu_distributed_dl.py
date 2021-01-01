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
