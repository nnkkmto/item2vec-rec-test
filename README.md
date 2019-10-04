# item2vec-food-rec

## 概要

以下論文が実際適用他のケースでも、普通のword2vecによるレコメンドよりも有効なのか検証

※ 普通のword2vecによるレコメンド
- sequenceをユニークにしない
- window数を全てのsequence対象にするようにしない

https://arxiv.org/abs/1603.04259

Item2Vec: Neural Item Embedding for Collaborative Filtering

### 想定ケース

論文内にもあるように、implicitかつセッション単位のデータにおける、item to itemのレコメンデーションを想定

### 使用データ

以下の、Instacartでのユーザー単位のオーダーデータを使用

上記のように、item2vecはセッションにおけるレコメーンデーションを想定しているため、

オーダーをセッションとして考え、ユーザー単位でのレコメンデーションは行わない

https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2

> “The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 2019/08/13

## セットアップ

### データ準備

以下2つのデータをダウンロードし、data以下に.tar.gzのまま配置

instacart

https://www.instacart.com/datasets/grocery-shopping-2017

criteo

https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/

### 学習・テストデータ、モデル作成

`python train.py`

### マスタデータ作成、対象アイテム予測（定性評価用）

`python predict.py --item '対象アイテムID'`

※ マスタデータ作成はtrain.pyに入れるもしくは他ファイルで行うように変更予定

※ instacartのみ用意

### 定量評価

`python evaluate.py`

評価指標は以下のサイト参照

http://blog.brainpad.co.jp/entry/2017/08/25/140000#MAPMean-Average-Precision

#### MRRによる評価

##### 問題設定

テストデータにおいて、ユーザーがアイテムをカートに追加後、その次にカートに追加するアイテムを当てられたかどうか

#### MAPによる評価

##### 問題設定

テストデータにおいて、ユーザーが対象アイテムをオーダーした際に、同時にオーダーした商品（複数）を当てられたかどうか
