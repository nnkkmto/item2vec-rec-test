
import os
import tarfile

import pandas as pd


def initialize_data(tarfile_path, directory, filename):

    if not (os.path.isfile(os.path.join(directory, filename+'.csv')) and
            os.path.isfile(os.path.join(directory, filename+'_test.csv'))):
        unpack_tar_gz(tarfile_path, directory)
        make_data(directory, filename)


def unpack_tar_gz(tarfile_path, directory):

    with tarfile.open(tarfile_path, 'r:gz') as tf:
        tf.extractall(path=directory)


def make_data(directory, filename):

    data_dir = os.path.join(directory, 'instacart_2017_05_01')

    df_orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    df_products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    df_aisles = pd.read_csv(os.path.join(data_dir, 'aisles.csv'))
    df_departments = pd.read_csv(os.path.join(data_dir, 'departments.csv'))

    df_order_products_prior = pd.read_csv(os.path.join(data_dir, 'order_products__prior.csv'))
    df_orders_prior = df_orders[df_orders.eval_set == 'prior']
    df_orders_prior = pd.merge(df_orders_prior, df_order_products_prior, on='order_id', how='inner')

    df_order_products_train = pd.read_csv(os.path.join(data_dir, 'order_products__train.csv'))
    df_orders_train = df_orders[df_orders.eval_set == 'train']
    df_orders_train = pd.merge(df_orders_train, df_order_products_train, on='order_id', how='inner')

    df = pd.concat([df_orders_prior, df_orders_train])

    df = pd.merge(df, df_products, on='product_id', how='inner')
    df = pd.merge(df, df_aisles, on='aisle_id', how='inner')
    df = pd.merge(df, df_departments, on='department_id', how='inner')

    df_train = df[df.eval_set == 'prior']
    df_test = df[df.eval_set == 'train']

    df_train.to_csv(os.path.join(directory, filename+'.csv'))
    df_test.to_csv(os.path.join(directory, filename+'_test.csv'))


def initialize_master(data_path, master_path, columns):

    if not os.path.isfile(master_path):
        make_master(data_path, master_path, columns)


def make_master(data_path, master_path, columns):
    df = pd.read_csv(data_path)
    df = df[columns]
    df = df.drop_duplicates()

    df.to_csv(master_path)




