
import pandas as pd
from gensim.models import word2vec

from lib import data_handler

def prepare_data():

    filepath = 'data/unpacked/data_test.csv'
    columns = ['order_id', 'order_number', 'add_to_cart_order', 'product_id']

    df = data_handler.load_data(filepath, columns)

    df = df.sort_values(by=['order_id', 'order_number', 'add_to_cart_order'])
    df = df.rename(columns={'product_id':'label'})
    df['product_id'] = df['label'].shift(1)
    df = df[df.add_to_cart_order != '1']
    df_label = df[['product_id', 'label']]

    item_list = df_label.product_id.unique()

    return df_label, item_list


def predict(model, item_list):

    df = pd.DataFrame(columns=['product_id', 'label', 'distance'])

    for item in item_list:
        try:
            result = model.wv.most_similar(item)
        except Exception:
            pass
        df_pred = pd.DataFrame(result, columns=['label', 'distance'])
        df_pred['product_id'] = item
        df_pred['rank'] = df_pred['distance'].rank(ascending=False)

        df = pd.concat([df, df_pred])

    return df


def evaluate_mrr(df_label, df_pred):

    df = pd.merge(df_label, df_pred, on=['product_id', 'label'], how='left')
    print(df)
    n = len(df)

    df_calc = df[~df.rank.isnull()]
    print(df_calc)
    df_calc['calc'] = 1 / df.rank
    print(df_calc)
    rr = df_calc.calc.sum()
    print(rr)
    print(n)
    print(rr / n)




def main():

    model = word2vec.Word2Vec.load('model/i2v.model')

    df_label, item_list = prepare_data()
    df_pred = predict(model, item_list)

    evaluate(df_label, df_pred)


if __name__ == '__main__':
    main()


