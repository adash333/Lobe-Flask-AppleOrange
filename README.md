# Lobe-Flask-AppleOrange

Microsoftの機械学習アプリLobe(beta版)を用いてリンゴとみかんの分類をFlaskアプリとしてHerokuにデプロイしたものです。

## デプロイしたもの

https://blooming-headland-09967.herokuapp.com/

## Microsoftの機械学習アプリLobe(beta版)でリンゴとみかんを分類するWEBアプリ作成を試してみる 目次

[（１）LobeのインストールからTensorFlowモデルのエクスポートまで](https://i-doctor.sakura.ne.jp/font/?p=44635)

[（２）Windows10でPython3.6+TensorFlow1.15をセットアップ](https://i-doctor.sakura.ne.jp/font/?p=44703)

[（３）Windows10ローカル環境でtf_example.pyを実行](https://i-doctor.sakura.ne.jp/font/?p=44808)

[（４）Windows10ローカル環境でFlaskを用いて画像判定](https://i-doctor.sakura.ne.jp/font/?p=44883)

[（５）FlaskアプリをHerokuにデプロイ](https://i-doctor.sakura.ne.jp/font/?p=44947)

## 実行方法

`git clone https://github.com/adash333/Lobe-Flask-AppleOrange.git`

Run `pipenv install`

Run `python app.py` to predict.(ローカル環境)

herokuへのデプロイに関しては、[Microsoftの機械学習アプリLobe(beta版)でリンゴとみかんを分類するWEBアプリ作成を試してみる（５）FlaskアプリをHerokuにデプロイ](https://i-doctor.sakura.ne.jp/font/?p=44947)　をご覧ください。


## 開発環境

```
Windows 10 Pro
VisualStudioCode 1.50.1
Git for Windows v2.29.2
python 3.6
pipenv 2020.8.13
```
