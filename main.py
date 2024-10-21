'''
Title : 使用手工规则的重排候选模型
'''

import numpy as np
import cudf, gc
from src.file_manager import FileManager
from src.co_visitation_matrix import CoVisitationMatrix
from src.handcrafted_rules import HandCraftedRules

VER = 1  # 版本号
print(f'Version {VER}\nRAPIDS version {cudf.__version__}')
if __name__ == '__main__':
    file_manager = FileManager()
    co_visitation_matrix = CoVisitationMatrix(file_manager)
    handcrafted_rules = HandCraftedRules()

    # # 读取文件、预处理
    # file_manager.read()
    # print(
    #     f'We will process {file_manager.files_len} files, in groups of {file_manager.READ_CT} and chunks of {file_manager.CHUNK}.')
    #
    # # 特征工程、计算共现矩阵（Co-visitation Matrix）
    # co_visitation_matrix.train()
    #
    # file_manager.clear_cache()
    # gc.collect()

    # 使用共现矩阵预测
    # co_visitation_matrix.load_metrix()
    # handcrafted_rules.train(co_visitation_matrix)
    # handcrafted_rules.save()
