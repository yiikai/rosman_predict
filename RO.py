import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset_dir = ["./train.csv", "./store.csv"]


def load_data(dir):
    data = pd.read_csv(dir[0])
    store = pd.read_csv(dir[1])
    return data,store

def preProcess(dataset):
    pass
    
def run():
    dataset,storedata = load_data(Dataset_dir)
    #merger两个数据集
    megerd = dataset.merge(storedata,on="Store")
    #不要限制显示
    pd.options.display.max_columns = None
    store_type = megerd.groupby('StoreType')
    store_counts = store_type.size()
    print(store_counts.values)
    plt.bar(store_counts.index,store_counts.values)
    plt.show()
   
 
if __name__=="__main__":
    print("start Anaylaze and predict work!!!!!!")
    run()