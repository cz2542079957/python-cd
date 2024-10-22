import pandas as pd, numpy as np

def show_pred(path = "./submission.csv"):
    pred = pd.read_csv(path)
    split_data  = pred["session_type"].str.split("_", expand=True)
    split_data.columns = ["session", "type"]
    pred[["session", "type"]] = split_data
    pred.drop("session_type", axis=1, inplace=True)
    pred["session"] = pred["session"].astype(int)
    pred.sort_values("session", inplace=True)

    # 随机抽取10个人查看
    session_max = int(pred["session"].max())
    session_min = int(pred["session"].min())
    random_index = set(np.random.randint(session_min, session_max, 10))
    cond_index = ((pred["session"].isin(random_index)) & (pred["type"] != "carts"))
    selected = pred[cond_index]
    selected.reset_index(drop=True, inplace=True)

    for session in set(selected.session):
        row = selected[selected["session"] == session]
        clicks = row[row["type"] == "clicks"]
        orders = row[row["type"] == "orders"]
        print(f"Session号为{session}的用户：")
        print(f"    可能点击的商品序列为：{np.array(clicks["labels"])}")
        print(f"    可能购买的商品序列为：{np.array(orders["labels"])}")
        print(f"\n")