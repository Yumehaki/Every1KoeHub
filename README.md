# Every1KoeHub

AIVMXモデル用のHubです。

## 使い方

```bash
# venv を作成して有効化
python3 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt

# アプリを起動
python3 app.py # または python app.py
```

[uv](https://github.com/astral-sh/uv) を使う場合:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run app.py
```
