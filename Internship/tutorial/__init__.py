import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    #data/[dataset_name]_dataset.pyのモジュールをインポートする
    dataset_filename = "data."+dataset_name+"_dataset" #dataset_filename = data.dataset_name_dataset -> モジュールへの相対パス
    datasetlib=importlib.import_module(dataset_filename) #モジュールのインポート
    
    datset=None
    target_dataset_name = dataset_name.replace('_', '') + "_dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \ #lower():小文字に変換
            and issubclass(cls, BaseDataset):
                dataset = cls
        
        if dataset is None:
            raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    #データセットクラスの静的メソッド<modify_commandline_options>を返す
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    #オプションを指定してデータセットを作成する
    #この関数は、CustomDatasetDataLoaderクラスをラップする
    #これはこのパッケージとtrain.py/test.pyの間のメインインタフェースである
    
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    #マルチスレッドデータの読み込みを実行するDatasetクラスのラッパークラス
    
    def __init__(self, opt):
        """このクラスの初期化
        
        step1: [dataset_mode]の名前のデータセットインスタンスを作成する
        step2: マルチスレッドデータローダーを作成する
        """
        
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.Dataloader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.serial_batches,
            num_workers = int(opt.num_threads))
    
    def load_data(self):
        return self

    def __len__(self):
        #データセットのデータ数を返す
        return min(len(self.dataset), self.opt_max_dataset_size)

    def __iter__(self)
        #データのバッチを返す
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
