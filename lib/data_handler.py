
import pandas as pd


def load_data(filepath, columns):

    df = pd.read_csv(filepath)
    df = df[columns]
    df = df.fillna('').astype(str)

    return df


def load_master(filepath):

    df = pd.read_csv(filepath)
    df = df.fillna('').astype(str)

    return df


def make_sentence(df, sort_columns, group_col, seq_col, seq_min_count):

    df = df.sort_values(by=sort_columns)
    df = df[[group_col, seq_col]]
    df = df.groupby(group_col)[seq_col].agg(list).reset_index()

    df_unique = df.copy()

    # ユニークなリストに変換
    df_unique[seq_col] = df_unique[seq_col].apply(lambda x: list(set(x)))

    # seq_min_count以下のユーザーを切り捨て
    df = df[df[seq_col].apply(len) >= seq_min_count]
    df_unique = df_unique[df_unique[seq_col].apply(len) >= seq_min_count]

    sentence_sequence = df[seq_col].values.tolist()
    sentence_set = df_unique[seq_col].values.tolist()

    return sentence_sequence, sentence_set






