import gc, cudf
import pandas as pd

from src.file_manager import FileManager

'''
构建三种共现矩阵，用于反映用户的兴趣热点与三种特征的关系，从而更好地理解用户行为模式
'''


class CoVisitationMatrix:
    DISK_PIECES_CARTS_ORDERS = 20
    DISK_PIECES_BUY2BUY = 4
    DISK_PIECES_CLICKS = 20

    top_20_buys = None
    top_20_buy2buy = None
    top_20_clicks = None

    top_clicks = None
    top_orders = None

    test_df = None

    def __init__(self, file_manager: FileManager, output_dir="./handled_files"):
        self.fm = file_manager
        self.output_dir = output_dir

    # 关注商品之间的共现关系（对三种行为分配不同权重，时间限度为1天）
    def carts_orders(self, disk_pieces=DISK_PIECES_CARTS_ORDERS):
        type_weight = {0: 1, 1: 6, 2: 3}
        # disk_pieces为对商品id的分片，意味着每次对这一范围内的，disk_pieces过小可能会导致爆显存
        # 注意最外层的分片是为了在循环末尾筛选该商品id范围内的商品，属于是逻辑分片
        size = 1.86e6 / disk_pieces  # 按照商品数划分，每一片有多少商品数
        # 按商品划分的片，逐片对数据进行处理
        for piece in range(disk_pieces):
            print('\n### DISK PART', piece + 1)

            # => OUTER CHUNKS
            # 第二层循环是物理分片，每次20个文件
            # 注意CHUNK和READ_CT的关系，READ_CT是为了减少爆显存风险，单次计算的文件数，CHUNK被分为了多个READ_CT小分片
            # 每次对计算完的所有READ_CT进行汇总到tmp2，再对所有CHUNK计算结果汇总到tmp
            tmp, tmp2, df = None, None, None
            for j in range(6):
                begin = j * self.fm.CHUNK
                end = min((j + 1) * self.fm.CHUNK, self.fm.files_len)
                print(f'Processing files {begin} ~ {end - 1} in groups of {self.fm.READ_CT}...')

                # => INNER CHUNKS
                # 第三、四层循环是进一步分片，每次对一个CHUNK读取READ_CT个文件再计算
                for k in range(begin, end, self.fm.READ_CT):
                    # 读取READ_CT个文件
                    df = [self.fm.read_file(self.fm.files[k])]
                    for i in range(1, self.fm.READ_CT):
                        if k + i < end: df.append(self.fm.read_file(self.fm.files[k + i]))
                    # 根据session ts排序
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(['session', 'ts'], ascending=[True, False])

                    # 对于每个session 只取30条（cumcount是累积计数）
                    df = df.reset_index(drop=True)
                    df['n'] = df.groupby('session').cumcount()
                    df = df.loc[df.n < 30].drop('n', axis=1)

                    # 以下都是关键操作
                    # 进行内联操作（共现分析的关键），通过对session进行内联，并且筛选出时间差在24小时以内、商品种类不同的一对操作
                    # 可以视为同一用户强相关的一对操作，这对后续预测非常关键
                    df = df.merge(df, on='session')
                    df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]

                    # 根据商品id进行逻辑分片
                    df = df.loc[(df.aid_x >= piece * size) & (df.aid_x < (piece + 1) * size)]

                    # 计算商品对的最终权重分布，越大说明关联性越强
                    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(
                        ['session', 'aid_x', 'aid_y'])  # 去重
                    df['wgt'] = df.type_y.map(type_weight)  # 根据类型分配权重
                    df = df[['aid_x', 'aid_y', 'wgt']]  # 保留主要属性
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x', 'aid_y']).wgt.sum()  # 对相同商品对求和，计算总权重

                    # 结合INNER CHUNKS计算结果
                    if k == begin:
                        tmp2 = df
                    else:
                        tmp2 = tmp2.add(df, fill_value=0)
                    print(k, ', ', end='')
                print()
                # 结合OUTER CHUNKS计算结果
                if begin == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)
                del tmp2, df
                gc.collect()
            # 根据 商品id 和 权重 排序
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
            # 只需要前15个
            tmp = tmp.reset_index(drop=True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp.loc[tmp.n < 15].drop('n', axis=1)
            # 保存结果
            tmp.to_pandas().to_parquet(f'{self.output_dir}/top_15_carts_orders_{piece}.pqt')
            del tmp
        return self

    # 关注购买行为的共现关系（它只关注加购物车和购买两种行为，并且时间限度为14天）
    def buy_2_buy(self, disk_pieces=DISK_PIECES_BUY2BUY):
        # disk_pieces为对商品id的分片，意味着每次对这一范围内的，disk_pieces过小可能会导致爆显存
        # 注意最外层的分片是为了在循环末尾筛选该商品id范围内的商品，属于是逻辑分片
        size = 1.86e6 / disk_pieces  # 按照商品数划分，每一片有多少商品数
        # 按商品划分的片，逐片对数据进行处理
        for piece in range(disk_pieces):
            print('\n### DISK PART', piece + 1)

            # => OUTER CHUNKS
            # 第二层循环是物理分片，每次20个文件
            # 注意CHUNK和READ_CT的关系，READ_CT是为了减少爆显存风险，单次计算的文件数，CHUNK被分为了多个READ_CT小分片
            # 每次对计算完的所有READ_CT进行汇总到tmp2，再对所有CHUNK计算结果汇总到tmp
            tmp, tmp2, df = None, None, None
            for j in range(6):
                begin = j * self.fm.CHUNK
                end = min((j + 1) * self.fm.CHUNK, self.fm.files_len)
                print(f'Processing files {begin} ~ {end - 1} in groups of {self.fm.READ_CT}...')

                # => INNER CHUNKS
                # 第三、四层循环是进一步分片，每次对一个CHUNK读取READ_CT个文件再计算
                for k in range(begin, end, self.fm.READ_CT):
                    # 读取READ_CT个文件
                    df = [self.fm.read_file(self.fm.files[k])]
                    for i in range(1, self.fm.READ_CT):
                        if k + i < end: df.append(self.fm.read_file(self.fm.files[k + i]))
                    # 根据session ts排序
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.loc[df['type'].isin([1, 2])]  # ONLY WANT CARTS AND ORDERS
                    df = df.sort_values(['session', 'ts'], ascending=[True, False])

                    # 对于每个session 只取30条（cumcount是累积计数）
                    df = df.reset_index(drop=True)
                    df['n'] = df.groupby('session').cumcount()
                    df = df.loc[df.n < 30].drop('n', axis=1)

                    # 以下都是关键操作
                    # 进行内联操作（共现分析的关键），通过对session进行内联，并且筛选出时间差在14天以内、商品种类不同的一对操作
                    # 可以视为同一用户强相关的一对操作，这对后续预测非常关键
                    df = df.merge(df, on='session')
                    df = df.loc[((df.ts_x - df.ts_y).abs() < 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y)]  # 14 DAYS

                    # 根据商品id进行逻辑分片
                    df = df.loc[(df.aid_x >= piece * size) & (df.aid_x < (piece + 1) * size)]

                    # 计算商品对的最终权重分布，越大说明关联性越强
                    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(
                        ['session', 'aid_x', 'aid_y'])  # 去重
                    df['wgt'] = 1  # 权重定为1
                    df = df[['aid_x', 'aid_y', 'wgt']]  # 保留主要属性
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x', 'aid_y']).wgt.sum()  # 对相同商品对求和，计算总权重

                    # 结合INNER CHUNKS计算结果
                    if k == begin:
                        tmp2 = df
                    else:
                        tmp2 = tmp2.add(df, fill_value=0)
                    print(k, ', ', end='')
                print()
                # 结合OUTER CHUNKS计算结果
                if begin == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)
                del tmp2, df
                gc.collect()
            # 根据 商品id 和 权重 排序
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
            # 只需要前15个
            tmp = tmp.reset_index(drop=True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp.loc[tmp.n < 15].drop('n', axis=1)
            # 保存结果
            tmp.to_pandas().to_parquet(f'{self.output_dir}/top_15_buy2buy_{piece}.pqt')
            del tmp
        return self

    # 关注用户浏览行为的共现关系（它分配的权重为时间，所以体现出用户浏览的即时兴趣和趋势，时间限度为1天）
    def clicks(self, disk_pieces=DISK_PIECES_CLICKS):
        # disk_pieces为对商品id的分片，意味着每次对这一范围内的，disk_pieces过小可能会导致爆显存
        # 注意最外层的分片是为了在循环末尾筛选该商品id范围内的商品，属于是逻辑分片
        size = 1.86e6 / disk_pieces  # 按照商品数划分，每一片有多少商品数
        # 按商品划分的片，逐片对数据进行处理
        for piece in range(disk_pieces):
            print('\n### DISK PART', piece + 1)

            # => OUTER CHUNKS
            # 第二层循环是物理分片，每次20个文件
            # 注意CHUNK和READ_CT的关系，READ_CT是为了减少爆显存风险，单次计算的文件数，CHUNK被分为了多个READ_CT小分片
            # 每次对计算完的所有READ_CT进行汇总到tmp2，再对所有CHUNK计算结果汇总到tmp
            tmp, tmp2, df = None, None, None
            for j in range(6):
                begin = j * self.fm.CHUNK
                end = min((j + 1) * self.fm.CHUNK, self.fm.files_len)
                print(f'Processing files {begin} ~ {end - 1} in groups of {self.fm.READ_CT}...')

                # => INNER CHUNKS
                # 第三、四层循环是进一步分片，每次对一个CHUNK读取READ_CT个文件再计算
                for k in range(begin, end, self.fm.READ_CT):
                    # 读取READ_CT个文件
                    df = [self.fm.read_file(self.fm.files[k])]
                    for i in range(1, self.fm.READ_CT):
                        if k + i < end: df.append(self.fm.read_file(self.fm.files[k + i]))
                    # 根据session ts排序
                    df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(['session', 'ts'], ascending=[True, False])

                    # 对于每个session 只取30条（cumcount是累积计数）
                    df = df.reset_index(drop=True)
                    df['n'] = df.groupby('session').cumcount()
                    df = df.loc[df.n < 30].drop('n', axis=1)

                    # 以下都是关键操作
                    # 进行内联操作（共现分析的关键），通过对session进行内联，并且筛选出时间差在24小时以内、商品种类不同的一对操作
                    # 可以视为同一用户强相关的一对操作，这对后续预测非常关键
                    df = df.merge(df, on='session')
                    df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]

                    # 根据商品id进行逻辑分片
                    df = df.loc[(df.aid_x >= piece * size) & (df.aid_x < (piece + 1) * size)]

                    # 计算商品对的最终权重分布，越大说明关联性越强
                    df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                    df['wgt'] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
                    df = df[['aid_x', 'aid_y', 'wgt']]  # 保留主要属性
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x', 'aid_y']).wgt.sum()  # 对相同商品对求和，计算总权重

                    # 结合INNER CHUNKS计算结果
                    if k == begin:
                        tmp2 = df
                    else:
                        tmp2 = tmp2.add(df, fill_value=0)
                    print(k, ', ', end='')
                print()
                # 结合OUTER CHUNKS计算结果
                if begin == 0:
                    tmp = tmp2
                else:
                    tmp = tmp.add(tmp2, fill_value=0)
                del tmp2, df
                gc.collect()
            # 根据 商品id 和 权重 排序
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
            # 只需要前20个
            tmp = tmp.reset_index(drop=True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp.loc[tmp.n < 20].drop('n', axis=1)
            # 保存结果
            tmp.to_pandas().to_parquet(f'{self.output_dir}/top_20_clicks_{piece}.pqt')
            del tmp
        return self

    def train(self):
        self.carts_orders().buy_2_buy().clicks()
        return self

    def load_metrix(self):
        self.top_20_buys = self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_15_carts_orders_0.pqt'))
        for k in range(1, self.DISK_PIECES_CARTS_ORDERS):
            self.top_20_buys.update(
                self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_15_carts_orders_{k}.pqt'))
            )

        self.top_20_buy2buy = self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_15_buy2buy_0.pqt'))
        for k in range(1, self.DISK_PIECES_BUY2BUY):
            self.top_20_buy2buy.update(
                self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_15_buy2buy_{k}.pqt'))
            )

        self.top_20_clicks = self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_20_clicks_0.pqt'))
        for k in range(1, self.DISK_PIECES_CLICKS):
            self.top_20_clicks.update(
                self.fm.pqt_to_dict(pd.read_parquet(f'{self.output_dir}/top_20_clicks_{k}.pqt'))
            )

        self.test_df = self.fm.load_test()
        print('Test data has shape', self.test_df.shape)

        self.top_clicks = self.test_df.loc[self.test_df['type'] == 'clicks', 'aid'].value_counts().index.values[:20]
        self.top_orders = self.test_df.loc[self.test_df['type'] == 'orders', 'aid'].value_counts().index.values[:20]

        print('Here are size of our 3 co-visitation matrices:')
        print(len(self.top_20_clicks), len(self.top_20_buy2buy), len(self.top_20_buys))
