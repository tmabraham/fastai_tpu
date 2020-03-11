from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from torchvision import models

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
    learn.fit(3)
    
if __name__ == "__main__":
    train_loop()