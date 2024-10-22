import glob, cudf
import numpy as np, pandas as pd

from utils.run_time import run_time


class FileManager:
    '''
     用于读取分片的parquet格式原始数据，并且建立映射表，供外部按照文件名获取DataFrame
    '''

    def __init__(self, root="./parquet"):
        self.data_cache = {}  # 缓存的文件字典 （key：文件名， val:DataFrame）
        self.type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}  # 标签映射表
        self.root = root
        self.files = glob.glob(f'{self.root}/*_parquet/*')
        self.files_len = len(self.files)
        self.READ_CT = 3
        self.CHUNK = int(np.ceil(self.files_len / 6))

    # 读入
    @run_time
    def read(self):
        print("开始建立文件字典")
        for f in self.files: self.data_cache[f] = self.__read_file_to_cache(f)
        print("文件字典建立完毕")

    # 用于根据路径读取parquet文件
    def __read_file_to_cache(self, filename):
        df = pd.read_parquet(filename)
        df.ts = (df.ts / 1000).astype('int32')
        df['type'] = df['type'].map(self.type_labels).astype('int8')
        return df

    # 用于直接从data_cache文件缓存映射表中读取数据
    def read_file(self, filename):
        return cudf.DataFrame(self.data_cache[filename])

    # 清理
    def clear_cache(self):
        self.data_cache = {}

    # 单独加载测试集
    def load_test(self):
        dfs = []
        for e, chunk_file in enumerate(glob.glob(f'{self.root}/test_parquet/*')):
            chunk = pd.read_parquet(chunk_file)
            chunk.ts = (chunk.ts / 1000).astype('int32')
            chunk['type'] = chunk['type'].map(self.type_labels).astype('int8')
            dfs.append(chunk)
        return pd.concat(dfs).reset_index(drop=True)  # .astype({"ts": "datetime64[ms]"})

    # pqt转换为dict
    def pqt_to_dict(self, df):
        return df.groupby('aid_x').aid_y.apply(list).to_dict()

    def load_predicted(self):
        return pd.read_csv("./submission.csv")