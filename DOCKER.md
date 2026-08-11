# Docker で動かす

このリポジトリは TensorFlow 1.14 / Python 3.6 / CUDA 10.0 という 2019 年当時の組み合わせに固定されているため、
再現用の Docker 環境を用意しています。Unity 製の環境バイナリ (`AnimalAI.x86_64`) も同じコンテナ内で動きます。

動作確認済みの構成: RTX 2080 Ti / ドライバ 580.178.04 / Docker 28.1.1。

## 1. 前提: NVIDIA Container Toolkit

ホスト側に NVIDIA ドライバと NVIDIA Container Toolkit が必要です。

```bash
nvidia-ctk --version                      # 入っているか確認
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtimes            # nvidia が出れば OK
```

## 2. ビルド

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose build
```

`USER_ID` / `GROUP_ID` はコンテナ内のユーザに使われます。省略すると 1000:1000 になります。
これを合わせておくと、bind mount した `runs/` や `nn/` に書かれるファイルがホスト側で root 所有になりません。

## 3. 動作確認

```bash
docker compose run --rm animal python docker/smoke_test.py
```

TensorFlow から GPU が見えること、Unity 環境が起動して 50 ステップ進むことを確認します。
最後に `smoke test OK` が出れば成功です。

## 4. 使い方

```bash
# 対話シェル
docker compose run --rm animal

# ネットワークのダウンロード (README のステップ 2)
docker compose run --rm animal python download_networks.py

# 任意のスクリプト
docker compose run --rm animal python your_script.py

# ノートブック (Player.ipynb / Validation.ipynb / test_a2c.ipynb)
docker compose up jupyter        # http://localhost:8888 (トークンなし)

# 学習曲線
docker compose up tensorboard    # http://localhost:6006
```

リポジトリは `/workspace` に bind mount されるので、ホスト側での編集はそのまま反映されます。

## 確認済みの動作

この環境で実際に通したもの:

- TensorFlow 1.14 から RTX 2080 Ti が見える (`Created TensorFlow device ... compute capability: 7.5`)
- Unity 環境の起動とステップ実行 (`docker/smoke_test.py`)
- ray 経由の並列環境 (`RayVecEnv`, 4 並列で約 150 env-steps/s)
- 学習済みネットワーク `nn/last84_10_5` を読み込んでの推論 (`Validation.ipynb` の内容)
- `test_a2c.ipynb` と同じ学習ループ (8 actors、2 エポック、約 200 FPS でチェックポイント保存まで)

## 中で何が起きているか

| 項目 | 内容 |
| --- | --- |
| ベース | `ubuntu:18.04` (Python 3.6 が標準で入るため) |
| CUDA | NVIDIA の deb アーカイブから CUDA 10.0 ランタイム + cuDNN 7 のみ取得。`nvidia/cuda:10.0-*` イメージは Docker Hub から削除済みで使えない |
| Python | `requirements.txt` をそのまま pin 通りにインストール (pip 21.3.1 が Python 3.6 対応の最終版) |
| 画面 | Xvfb を `:99` で起動 (`docker/entrypoint.sh`) |
| OpenGL | VirtualGL の EGL バックエンド経由で NVIDIA ドライバに描画させる |

### なぜ VirtualGL が必要か

Xvfb のソフトウェア OpenGL は core profile を出せず、Unity 2018.3 のプレイヤーは
`Unable to find a supported OpenGL core profile` で終了します。
そこで entrypoint がコマンドを `vglrun -d egl0` 経由で起動し、描画は EGL で NVIDIA ドライバに投げ、
結果を Xvfb に転送します。これで Unity 側は `Renderer: NVIDIA GeForce RTX 2080 Ti` を掴んで起動します。

`vglrun` は `LD_PRELOAD` を設定して子プロセスに引き継がせるので、Python から `subprocess` で起動される
Unity プロセスにも自動的に効きます。

## 既知の注意点

- **`hyperparams.py` の `BASE_DIR`** はコンテナ内のパス `/workspace` に変更してあります。ホスト側で直接動かす場合は書き換えてください。
- **`LEARNING_DIR`** は元は `configs/learning/competition_configurations/` でしたが、このディレクトリはリポジトリに含まれていません
  (そのままだと `RayVecEnv` のワーカーが起動時に `FileNotFoundError` になります)。`hyperparams.py` 冒頭のコメント
  「For first submitted network with score 42.66 I used stage3」に従い `configs/learning/stage3/` を指すようにしています。
- **GPU メモリ**: `games_configurations.py` の `NUM_ACTORS` は 24 で、Unity プロセス 1 つごとに GPU メモリを消費します。
  ノートブック側で `per_process_gpu_memory_fraction=0.8` を指定しているため、11GB クラスの GPU では OOM になる可能性があります。
  その場合は `NUM_ACTORS` を減らすか、`per_process_gpu_memory_fraction` を下げてください (README も同様の注意を書いています)。
- **`download_networks.py`**: Google Drive の仕様変更で壊れていたため修正しました (旧 `docs.google.com/uc` + `download_warning`
  クッキー方式は HTML の確認ページを返すだけで、`networks.zip` が 2.4KB の HTML になっていました)。現在の
  `drive.usercontent.google.com` のフォーム送信方式に合わせてあります。
- **ALSA / FMOD のエラーログ**: コンテナにサウンドデバイスがないために出るもので、動作に影響しません。
- **ray 終了時の `get_global_worker` AttributeError**: ray 0.6.6 のシャットダウン時のノイズで、無視して問題ありません。
- **GPU なしでの実行**: TensorFlow は CPU にフォールバックしますが、Unity 環境は起動できません (VirtualGL が使えないため)。
