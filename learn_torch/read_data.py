# 使用 DeepL 环境
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):

  def __init__(self,root_dir,label_dir):
    self.root_dir = root_dir
    self.label_dir = label_dir
    self.path = os.path.join(self.root_dir,self.label_dir)
    self.img_path = os.listdir(self.path)

  def __getitem__(self, index):
    img_name = self.img_path[index]
    img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
    img = Image.open(img_item_path)
    label = self.label_dir
    return img,label
  
  def __len__(self):
    return len(self.img_path)
  

root_dir = r"..\数据集\hymenoptera_data\train"
ant_dir = "ants"
bee_dir = "bees"
ant_dataset = MyDataset(root_dir,ant_dir)
bee_dataset = MyDataset(root_dir,bee_dir)

train_dataset = ant_dataset + bee_dataset
  
