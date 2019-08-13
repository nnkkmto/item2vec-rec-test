
import os
import logging

from gensim.models import word2vec

from lib import data_handler
from lib import data_initializer


def prepare_data():

    tarfile_path = 'data/instacart_online_grocery_shopping_2017_05_01.tar.gz'
    data_dir = 'data/unpacked'
    data_filename = 'data'

    columns = ['user_id', 'order_id', 'order_number', 'add_to_cart_order', 'product_id']

    sort_columns = ['user_id', 'order_number', 'add_to_cart_order']
    group_col = 'order_id'
    seq_col = 'product_id'
    seq_min_count = 2

    print('initializing data')
    data_initializer.initialize_data(tarfile_path, data_dir, data_filename)

    print('loading data')
    df = data_handler.load_data(os.path.join(data_dir, data_filename+'.csv'), columns)

    print('making sentences')
    sentence_sequence, sentence_set = data_handler.make_sentence(
        df, sort_columns, group_col, seq_col, seq_min_count)

    return sentence_sequence, sentence_set


def train(sentences, is_max_window):

    if is_max_window:
        # window数は単語数の最大+1にする
        window = len(max(sentences, key=len)) + 1
    else:
        window = 5

    model = word2vec.Word2Vec(
        size=40,
        min_count=10,
        sg=1,
        hs=0,
        negative=15,
        sample=1e-2,
        iter=5,
        window=window
        )

    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    return model


def main():

    sentence_sequence, sentence_set = prepare_data()

    print('training simple w2v')
    model_w2v = train(sentence_sequence, is_max_window=False)
    model_w2v.save('model/simple_w2v.model')

    print('training max window w2v')
    model_w2v_max_window = train(sentence_sequence, is_max_window=True)
    model_w2v_max_window.save('model/max_window_w2v.model')

    print('training i2v')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model_i2v = train(sentence_set, is_max_window=True)
    model_i2v.save('model/i2v.model')


if __name__ == '__main__':
    main()
