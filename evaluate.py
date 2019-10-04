
import os

import pandas as pd
from gensim.models import word2vec

from lib import data_handler

def prepare_data_instacart():

    filepath = 'data/instacart/unpacked/data_test.csv'
    columns = ['order_id', 'order_number', 'add_to_cart_order', 'product_id']

    df = data_handler.load_data(filepath, columns)
    item_list = df.product_id.unique()

    df_label_mrr = df.sort_values(by=['order_id', 'order_number', 'add_to_cart_order'])
    df_label_mrr = df_label_mrr.rename(columns={'product_id':'label'})
    df_label_mrr['product_id'] = df_label_mrr['label'].shift(1)
    df_label_mrr = df_label_mrr[df_label_mrr.add_to_cart_order != '1']
    df_label_mrr = df_label_mrr[['product_id', 'label']]

    df_label_map = df.rename(columns={'product_id': 'label'})
    df_label_map = df_label_map[['order_id', 'label']]
    df_label_map = pd.merge(df[['order_id', 'product_id']], df_label_map, on='order_id', how='left')
    df_label_map = df_label_map[df_label_map.product_id != df_label_map.label]

    return df_label_mrr, df_label_map, item_list


def prepare_data_criteo():

    filepath = 'data/criteo/unpacked/data_test.csv'
    columns = ['click_timestamp', 'product_id', 'user_id']

    df = data_handler.load_data(filepath, columns)
    print(df)

    df = df.sort_values(by=['click_timestamp']).reset_index()
    df['time_sort'] = df.index
    df['order_id'] = df.groupby(by='user_id')['time_sort'].rank(method='first')

    item_list = df.product_id.unique()

    df_label_mrr = df.rename(columns={'product_id': 'label'})
    df_label_mrr['product_id'] = df_label_mrr['label'].shift(1)
    df_label_mrr = df_label_mrr[df_label_mrr.order_id != '1']
    df_label_mrr = df_label_mrr[['product_id', 'label']]

    df_label_map = df.rename(columns={'product_id': 'label'})
    df_label_map = df_label_map[['user_id', 'label']]
    df_label_map = pd.merge(df[['user_id', 'product_id']], df_label_map, on='user_id', how='left')
    df_label_map = df_label_map[df_label_map.product_id != df_label_map.label]

    return df_label_mrr, df_label_map, item_list



def predict(model, item_list):

    df = pd.DataFrame(columns=['product_id', 'label', 'distance'])

    for item in item_list:
        try:
            result = model.wv.most_similar(item)
            df_pred = pd.DataFrame(result, columns=['label', 'distance'])
            df_pred['product_id'] = item
            df_pred['rank'] = df_pred['distance'].rank(ascending=False)

            df = pd.concat([df, df_pred], sort=True)
        except Exception:
            pass

    return df


def evaluate_mrr(df_label, df_pred):

    df = pd.merge(df_label, df_pred, on=['product_id', 'label'], how='left')

    n = len(df)

    df_calc = df.dropna()
    df_calc['calc'] = 1 / df_calc['rank'].astype(int)
    rr = df_calc.calc.sum()
    mrr = rr / n

    print(mrr)


def evaluate_map(df_label, df_pred):

    # 最小単位はorder_id*product_idなので、修正
    df = pd.merge(df_label, df_pred, on=['product_id', 'label'], how='left')

    n = len(df[['order_id', 'product_id']].drop_duplicates())

    df = df.dropna()
    df['correct'] = df.groupby(by=['order_id', 'product_id'])['rank'].rank(ascending=True)
    df['ap'] = df.correct / df['rank'].astype(int)
    ap = df.ap.sum()

    map_metr = ap / n
    print(map_metr)


def main():

    # print('----- instacart -----')
    # df_label_mrr, df_label_map, item_list = prepare_data_instacart()
    #
    # # for model_name in ['simple_w2v', 'max_window_w2v', 'i2v']:
    # for model_name in ['simple_w2v', 'i2v']:
    #
    #     model_path = os.path.join('model/instacart/', model_name+'.model')
    #     model = word2vec.Word2Vec.load(model_path)
    #
    #     df_pred = predict(model, item_list)
    #
    #     print(f'MAP: {model_name}\n')
    #     evaluate_map(df_label_map, df_pred)
    #
    #     print(f'MRR: {model_name}\n')
    #     evaluate_mrr(df_label_mrr, df_pred)

    print('----- criteo -----')
    df_label_mrr, df_label_map, item_list = prepare_data_criteo()

    # for model_name in ['simple_w2v', 'max_window_w2v', 'i2v']:
    for model_name in ['simple_w2v', 'i2v']:
        model_path = os.path.join('model/criteo/', model_name + '.model')
        model = word2vec.Word2Vec.load(model_path)

        df_pred = predict(model, item_list)

        print(f'MAP: {model_name}\n')
        evaluate_map(df_label_map, df_pred)

        print(f'MRR: {model_name}\n')
        evaluate_mrr(df_label_mrr, df_pred)


if __name__ == '__main__':
    main()


