# item2vec-food-rec

## 概要

以下論文が実際適用他のケースでも、普通のword2vecによるレコメンドよりも有効なのか検証

※ 普通のword2vecによるレコメンド
- sequenceをユニークにしない
- window数を全てのsequence対象にするようにしない

https://arxiv.org/abs/1603.04259

Item2Vec: Neural Item Embedding for Collaborative Filtering

## セットアップ

### データ準備

以下のデータをダウンロードし、data以下に.tar.gzのまま配置

https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2

### 学習・テストデータ、モデル作成

`python train.py`

### マスタデータ作成、対象アイテム予測（定性評価用）

`python predict.py --item '対象アイテムID'`

※ マスタデータ作成はtrain.pyに入れるもしくは他ファイルで行うように変更予定

### 定量評価（MRR）

`python evaluate.py`

テストデータを元に、ユーザーがアイテムをカートに追加後、その次にカートに追加するアイテムを当てられたかどうかを問題設定とする

MRRにより定量評価

