import itertools
from collections import Counter
import numpy as np, pandas as pd

from src.co_visitation_matrix import CoVisitationMatrix


class HandCraftedRules:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    pred_df_clicks = None
    pred_df_buys = None

    def __init__(self):
        pass

    def suggest_clicks(self, df, cvm: CoVisitationMatrix):
        aids = df.aid.tolist()
        types = df.type.tolist()
        unique_aids = list(dict.fromkeys(aids[::-1]))  # 去重，并保持最近的aid在前面

        # 创建一个计数器来存储aid和它们的加权得分
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * self.type_weight_multipliers[t]
            sorted_aids = [k for k, v in aids_temp.most_common(20)]  # 获取得分最高的20个aid
            return sorted_aids

        # 从clicks共现矩阵中获取与unique_aids相关的aid，并展平列表
        aids2 = list(itertools.chain(*[cvm.top_20_clicks[aid] for aid in unique_aids if aid in cvm.top_20_clicks]))

        # 计数并排序，排除已在unique_aids中的aid
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]

        result = unique_aids + top_aids2[:20 - len(unique_aids)]  # 合并列表，确保结果长度为20
        return result + list(cvm.top_clicks)[:20 - len(result)]  # 如果结果不足20，用测试期间的点击补充

    def suggest_buys(self, df, cvm: CoVisitationMatrix):
        aids = df.aid.tolist()
        types = df.type.tolist()

        # 去重
        unique_aids = list(dict.fromkeys(aids[::-1]))
        df = df.loc[(df['type'] == 1) | (df['type'] == 2)]  # 筛选出加购物车和购买类型的数据
        unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

        # 创建一个计数器来存储aid和它们的加权得分
        if len(unique_aids) >= 20:
            weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * self.type_weight_multipliers[t]  # 根据类型权重乘数更新aid的得分
            # 从共现矩阵中获取与unique_buys相关的aid，并展平列表
            aids3 = list(
                itertools.chain(*[cvm.top_20_buy2buy[aid] for aid in unique_buys if aid in cvm.top_20_buy2buy]))

            for aid in aids3: aids_temp[aid] += 0.1  # 为这些aid增加额外的权重
            sorted_aids = [k for k, v in aids_temp.most_common(20)]  # 获取得分最高的20个aid
            return sorted_aids

        # 从buys共现矩阵中获取与unique_aids相关的aid，并展平列表
        aids2 = list(itertools.chain(*[cvm.top_20_buys[aid] for aid in unique_aids if aid in cvm.top_20_buys]))
        # 从buy2buy共现矩阵中获取与unique_buys相关的aid，并展平列表
        aids3 = list(itertools.chain(*[cvm.top_20_buy2buy[aid] for aid in unique_buys if aid in cvm.top_20_buy2buy]))

        # 计数并排序，排除已在unique_aids中的aid
        top_aids2 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(20) if aid2 not in unique_aids]

        result = unique_aids + top_aids2[:20 - len(unique_aids)]  # 合并列表，确保结果长度为20
        return result + list(cvm.top_orders)[:20 - len(result)]  # 如果结果不足20，用测试期间的点击补充

    def train(self, cvm: CoVisitationMatrix):
        self.pred_df_clicks = cvm.test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
            lambda x: self.suggest_clicks(x, cvm)
        )

        self.pred_df_buys = cvm.test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
            lambda x: self.suggest_buys(x, cvm)
        )

    def save(self):
        clicks_pred_df = pd.DataFrame(self.pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
        orders_pred_df = pd.DataFrame(self.pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
        carts_pred_df = pd.DataFrame(self.pred_df_buys.add_suffix("_carts"), columns=["labels"]).reset_index()
        pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
        pred_df.columns = ["session_type", "labels"]
        pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
        pred_df.to_csv("submission.csv", index=False)
        print("Saved submission to submission.csv\n", pred_df.head())
