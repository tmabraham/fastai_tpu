from torch.multiprocessing import Pool, set_start_method
from fastai.vision.all import *


def filelist2df(path):
    df = pd.read_csv(path, delimiter='/', header=None, names=['label', 'name'])
    df['name'] =  df['label'].astype(str) + "/" + df['name'].astype(str) + ".jpg"
    return df

path = untar_data(URLs.FOOD)
train_path = path/'train.txt'
test_path = path/'test.txt'

def load_data(index):
    train_df = filelist2df(train_path)
    test_df = filelist2df(test_path)
    food = DataBlock(blocks=(ImageBlock, CategoryBlock), get_x = ColReader(1,pref=path/'images'), splitter = RandomSplitter(), get_y = ColReader(cols=0), item_tfms=Resize(224))
    dls = food.dataloaders(train_df.values,bs=64)


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    try:
        pool = Pool(8)
        pool.map(load_data, [1,2,3,4,5,6,7,8])
    except KeyboardInterrupt:
        exit()
    finally:
        pool.terminate()
        pool.join()
