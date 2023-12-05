# OpenAI
このrepositoryは、[OpenAI API](https://platform.openai.com/docs/quickstart?context=python)の機能を利用し、個人的に扱いやすいようにパッケージングした物です。他のPythonパッケージでの利用なども可能です。

## 認証キー、OpenAI APIを取得する
[公式サイト](https://platform.openai.com/api-keys)から認証キーを取得し、環境変数として扱えるように設定します。
```bash
export OPENAI_API_KEY="123456..." # ご自身が取得した認証キーを記載
```
次にOpenAI APIをインストールする。
```bash
pip install --upgrade openai
```

## Clone, 利用
以下のコマンドから利用可能です。
```bash
git clone git@github.com:oishireiyo/OpenAI.git
cd OpenAI/src
python inputGPT4Vision.py
```
