
import os
import argparse

import pandas as pd
from gensim.models import word2vec

from lib import data_handler
from lib import data_initializer


def prepare_master():
    data_path = 'data/unpacked/data.csv'
    master_path = 'data/unpacked/master.csv'
    columns = ['product_id', 'product_name', 'aisle', 'department']

    data_initializer.initialize_master(data_path, master_path, columns)
    df_master = data_handler.load_master(master_path)

    return df_master


def load_models():
    model_dir = 'model/'
    model_names = ['simple_w2v', 'i2v']

    model_dict = {}
    for model_name in model_names:
        model_dict[model_name] = word2vec.Word2Vec.load(os.path.join(model_dir, model_name + '.model'))

    return model_dict


def predict(item_key, model_dict, df_master):

    print(f'対象item: {item_key} \n')

    for model_name, model in model_dict.items():

        result = model.wv.most_similar(item_key)
        df_pred = pd.DataFrame(result, columns=['product_id', 'distance'])
        df_pred = pd.merge(df_pred, df_master, on='product_id', how='left')

        print(f'予測モデル: {model_name}')
        print(df_pred)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--item', dest='item', type=str, required=True)
    args = parser.parse_args()

    return args


def main():

    args = get_args()

    df_master = prepare_master()
    model_dict = load_models()
    predict(args.item, model_dict, df_master)


if __name__ == '__main__':
    main()



