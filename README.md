# PAI SO-101 Final Project

Physical AI応用1講座の最終課題に向けて、SO-101実機とAIを組み合わせたロボット制御に取り組むプロジェクトです。

## 目的・課題テーマ（検討中）

SO-101のLeader / Followerを使用し、カメラ画像に対するAI推論の結果に応じて、Followerに物体の把持・移動・仕分けを実行させることを目指します。

- **基本目標**：AIを利用した物体認識と、LeRobotによる実機の動作制御を連携させる。
- **候補テーマ**：K20Proカメラを用いた物体の認識・自動仕分け（現時点では案）。
- **発展目標（任意）**：ACT等による模倣学習、ROS 2との連携、成功率などの定量評価。

最終課題では**何らかのAIの使用が必須**です。AI推論のみでもよく、新たなモデル学習は任意です。

## 開発環境

| 項目 | 環境 |
| --- | --- |
| OS | Windows 11 / Ubuntu 24.04 LTS（WSL2） |
| Python | 3.12（uv / `.venv`） |
| ロボット制御 | LeRobot 0.6.1 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| ロボット | SO-101 Leader / Follower |
| カメラ | HK 1080P K20Pro（USB） |
| データ形式 | LeRobotDataset（Parquet） |

## 開発状況（2026-10-12時点）

### 1. 開発環境の構築

- [x] Python 3.12 / uvの環境構築
- [x] LeRobot 0.6.1のインストール
- [x] RTX 5070 TiでのCUDA動作確認
- [x] Hugging Face CLI認証

### 2. SO-101の組み立て・基本操作

- [x] SO-101 Leader / Followerの組み立て
- [x] サーボモーターID設定
- [x] Leader / FollowerのキャリブレーションとJSON保存
- [x] `usbipd-win`によるWindowsからWSL2へのUSB接続
- [x] LeRobotによるLeader → Followerのテレオペレーション
- [x] 実機の基本動作確認
- [ ] テレオペレーション時の動作精度・安全性の追加検証

### 3. カメラ・データ収集

- [x] K20ProカメラをWSL2上で認識（`/dev/video0`）
- [x] V4L2で対応フォーマットを確認（例：640×480、30 fps）
- [x] テレオペレーションによる動作データの収集（カメラなし）
- [x] LeRobotDatasetの作成・保存内容の確認（1エピソード、898フレーム）
- [ ] `lerobot-find-cameras opencv`によるLeRobot側のカメラ検出
- [ ] K20Proカメラの設置位置を固定・撮影範囲を確認
- [ ] カメラ映像と関節データの同時記録
- [ ] 保存した関節データの可視化・分析

### 4. AIを利用したロボット制御（最終課題の中心）

- [ ] 課題タスク・対象物・成功条件を決定
- [ ] 使用するAIモデルの選定・動作確認
- [ ] カメラ画像に対するAI推論（物体認識など）
- [ ] AI推論結果を使った動作の選択・指令生成
- [ ] AI推論とLeRobotによるFollower制御の連携
- [ ] 実機での動作テスト・成功率等の評価

※ LeRobotによる**手動のテレオペレーション制御は完了**していますが、AI推論に基づく自律制御は未実装です。

### 5. 模倣学習（発展課題・任意）

- [x] テレオペレーションによる最初のデータ収集
- [x] LeRobotDatasetをローカルに作成
- [ ] カメラ付きの把持・移動データを複数エピソード収集
- [ ] ACT等の模倣学習モデルを学習
- [ ] 学習済みモデルによる推論・自律制御
- [ ] 必要に応じてHugging Faceへデータ・モデルを保存

### 6. ROS 2との連携（発展課題・任意）

- [ ] このプロジェクト向けROS 2環境の整備
- [ ] Publisher / Subscriberによるトピック通信
- [ ] AI推論結果のROS 2トピック配信
- [ ] ROS 2とLeRobotによるSO-101制御の連携
- [ ] ロボット動作状態の取得・配信

### 7. 最終課題提出

- [ ] GitHubリポジトリをpublicで公開し、READMEとコードを整備
- [ ] 1〜3分の実機デモ動画を作成
- [ ] 説明資料（PDF）を作成
- [ ] YouTubeに動画をアップロード（最低限、限定公開）
- [ ] omnicampusにPDF・リポジトリURL・動画URLを提出

## 実施済みの動作確認・記録

### キャリブレーション

両方ともLeRobotでキャリブレーションファイルの保存に成功しています。

- Follower：`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
- Leader：`~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json`

※ キャリブレーション保存済みでも、初回や再接続後の実機操作は小さい動作から確認します。

### 初回LeRobotDataset（2026-10-11）

- 保存先：`~/projects/so101/datasets/so101_teleop_001`
- エピソード数：**1**
- 総フレーム数：**898**
- 設定FPS：**30**
- 記録項目：`action`、`observation.state`、`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`
- カメラ：**未設定**（画像・動画なし）
- Hugging Face Hub：**未アップロード**（ローカル保存）

保存ファイルの例：

```text
datasets/so101_teleop_001/
├── data/chunk-000/file-000.parquet
└── meta/
    ├── episodes/chunk-000/file-000.parquet
    ├── info.json
    ├── stats.json
    └── tasks.parquet
```

記録中に処理速度が**26.5 Hz**まで低下した警告が一度出ています。次回のカメラ付き収録では処理負荷と記録周期を確認します。

### K20Proの認識状況

- WSL2のV4L2デバイス：`/dev/video0`（別機能として`/dev/video1`、`/dev/media0`も表示）
- 対応例：640×480 / 30 fps（YUYV・MJPG）、1280×720 / 30 fps（MJPG）
- **OS側では認識済み**。LeRobotのカメラ設定・撮影データの保存はこれから実施します。

## 操作メモ

仮想環境の起動：

```bash
cd ~/projects/so101
source .venv/bin/activate
```

USB通信ポートとカメラの確認：

```bash
ls -l /dev/ttyACM*
v4l2-ctl --list-devices
lerobot-find-cameras opencv
```

最後のテレオペレーション時のポート構成はFollower=`/dev/ttyACM0`、Leader=`/dev/ttyACM1`でした。**USB再接続後は番号が変わることがあるため、実行前に対応関係を再確認します。**

確認済みのテレオペレーションコマンド：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader
```

既存のデータセット情報の確認：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path.home() / "projects/so101/datasets/so101_teleop_001"
info = json.loads((root / "meta/info.json").read_text())
print("episodes:", info["total_episodes"])
print("frames:", info["total_frames"])
print("fps:", info["fps"])
print("features:", list(info["features"]))
PY
```

## 次に行うこと（優先順）

1. **K20ProをLeRobotに認識させる**：`lerobot-find-cameras opencv`の結果を確認する。
2. **カメラ付きの短いデータ収集**：新しいデータセット名で画像＋関節データを記録する。
3. **タスクを具体化する**：固定位置の物体を把持し、AIの分類結果で置き場所を切り替える構成を検討する。
4. **AIモデル・実機制御を連携させる**：まず小さな動作で安全を確認する。
5. **実験・提出物作成**：成功率・失敗事例をまとめ、動画とPDFで説明する。

## 注意事項

- 通電中にサーボ用3ピンケーブルを抜き差ししない。
- Followerの起動前は、Leaderと近い初期姿勢に合わせて急な動作を避ける。
- `Input voltage error!` が再発した場合は、繰り返し起動せず電源・接続を点検する。
- ローカル収録データや個人情報・秘密情報を、意図せず公開GitHubに含めない。
- 収録をやり直す場合は、保存済みの`so101_teleop_001`を上書きしないよう、新しいデータセット名を使用する。

## 最終課題の提出条件

- **提出締切：2026年11月2日（月）10:00**
- **GitHub**：publicのリポジトリURL（使用したコード一式）
- **動画**：1〜3分。YouTubeで最低限「限定公開」（倍速使用時は明記）
- **説明資料**：PDFで、目的・構成・AIの利用方法・工夫・結果・考察を説明
- **提出先**：omnicampus（PDF、GitHub URL、YouTube URL）

提出物には、氏名・所属・メールアドレスなどの個人情報を含めないようにします。


## main(提出物準備)
## AI Model

- Model: Google OWLv2
- Model ID: google/owlv2-base-patch16-ensemble
- Task: Zero-Shot Object Detection
- License: Apache License 2.0
- Source: https://huggingface.co/google/owlv2-base-patch16-ensemble