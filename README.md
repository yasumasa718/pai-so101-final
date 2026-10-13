# PAI SO-101 Final Project

Physical AI応用1講座の最終課題として、**OWLv2による潰した空き缶の認識**と、**SO-101による回収・白い箱への投入**を目指すプロジェクトです。

> **更新日：2026-10-13（模擬画像による画像・動画収録のオフライン検証を追加）**\
> **提出期限：2026-11-02（月）10:00（日本時間）**\
> **現在の到達点：Fake Robotと模擬画像によるエピソード収録、実LeRobotDatasetへの画像・動画保存と読戻し、起動時の校正・設定照合をオフライン検証済み。実機収録の安全確認、実カメラとの同時収録、OWLv2による自動回収は未完了。**

## 1. 目的と達成目標

### 最終課題の目標

1. K20Proカメラで潰した空き缶を撮影する。
2. Google OWLv2で対象物を検出し、ROI（画像上の関心領域）や信頼度で結果を検証する。
3. 検証済みの認識結果を、安全に確認された把持・投入動作の選択につなげる。
4. SO-101 Followerで空き缶を回収し、白い箱へ投入する。
5. 実機デモと成功／失敗事例を記録し、結果を提出する。

**重要：上記は目標であり、現時点で自動回収に成功したことを意味しません。** 検出画像のROIはロボットの到達範囲や把持位置を保証しません。画像座標とロボット座標の対応、衝突回避、アーム支持などは別途検証が必要です。

発展目標（任意）：ACT等による模倣学習、ROS 2との連携。提出期限までは基本目標を優先します。

## 2. 開発環境

| 項目 | 環境 |
| --- | --- |
| OS | Windows 11 + WSL2 Ubuntu 24.04 LTS |
| Python | Python 3.12（`uv` / `.venv`） |
| ロボット制御 | Hugging Face LeRobot 0.6.1 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| ロボット | SO-101 Leader / Follower |
| カメラ | HK 1080P K20Pro（USB、WSL2では `/dev/video0` を確認済み） |
| 画像認識 | Google OWLv2 |
| 収録形式 | LeRobotDataset v3形式（Parquet） |
| テスト | pytest |
| リポジトリ | [yasumasa718/pai-so101-final](https://github.com/yasumasa718/pai-so101-final) |
| 作業ブランチ | `dev` |

USBシリアルポートは `/dev/serial/by-id/` の機器固有パスを用いてFollowerとLeaderを識別済みです。公開用READMEには機器固有のUSBシリアル番号を記載しません。実際の接続先は実行前に確認します。

## 3. 開発状況（2026-10-13）

### 3.1 開発環境・基本操作

- [x] Python 3.12 / `uv` の環境構築
- [x] LeRobot 0.6.1の導入
- [x] RTX 5070 Ti / CUDAの動作確認
- [x] Hugging Faceの認証
- [x] SO-101 Leader / Followerの組み立て、モーターID設定
- [x] Leader / FollowerのキャリブレーションとJSON保存
- [x] WindowsからWSL2へのUSB接続
- [x] LeaderによるFollowerのテレオペレーション
- [ ] 実機の動作精度・機械的安全性・非常時の停止手順の追加検証

### 3.2 カメラとOWLv2

- [x] K20ProをWSL2上で認識、画像取得
- [x] V4L2による対応解像度・フレームレートの確認
- [x] OWLv2による空き缶の静止画像認識
- [x] ROI・信頼度による認識結果のオフライン検証
- [ ] 固定カメラの設置位置と視野の再確認
- [ ] LeRobotの収録処理でのカメラ同時記録を確認
- [ ] 画像座標から安全な把持／投入動作につなぐ処理

認識結果が `PASS` でも**ロボットへの動作許可にはなりません**。

### 3.3 データ収録・軌道検証

- [x] テレオペレーションによるLeRobotDataset収録の基本動作
- [x] `validate_trajectory.py` の実装とオフラインテスト
- [x] 既存データセットのF4設定範囲違反を検出
- [x] `DatasetRecordWriter` で `safe_record_loop.py` と実LeRobotDatasetの `add_frame()` 等をオフライン接続
- [x] `SafeRecordSession` で正常エピソードの保存・読戻しと異常エピソードの破棄をオフライン検証
- [x] ネイティブ保存開始後の失敗時に `commit_uncertain` を保持し、再保存・破棄・終了を禁止することを検証
- [x] 模擬RGB画像と送信目標指令の同時記録、PNG／H.264動画の保存・読戻しを検証
- [ ] `so101_pick_place_004` の新規収録（**未収録**）
- [ ] 004に対する軌道・データ整合性検証
- [ ] カメラ映像と関節データの同時収録を確認

### 3.4 Action Guard・安全な収録処理

- [x] `action_guard.py`：Followerキャリブレーション由来の6関節設定範囲を検証し、範囲外指令を補正せず拒否
- [x] `guarded_record_processor.py`：LeRobot Processorとのオフライン連携
- [x] `safe_record_loop.py`：1フレーム単位での指令検証・送信・保存コールバック
- [x] `safe_record_loop.py`：送信前の再検証、送信戻り値の保存、FAULT状態の保持
- [x] `dataset_record_writer.py`：観測の事前ステージング、送信戻り値のfloat32記録、明示的な保存・破棄・終了
- [x] `safe_record_session.py`：1エピソードの状態遷移とFAULT時の保存禁止を統括（オフライン統合済み）
- [x] `record_startup_check.py`：固定SHA-256・読込み済み校正・`use_degrees`・Guard範囲／単位／余裕幅の起動時照合（オフライン検証）
- [x] Fake Robotによる異常系テスト（実機・カメラI/Oなし）
- [x] **pytest全180件成功（2026-10-13、Python 3.12.3）**
- [x] 実LeRobotDatasetで複数フレーム・複数エピソードの保存、Parquet・メタデータ読戻しを検証（画像なし）
- [x] F2・F4の生指令／最終送信候補の違反、通信例外、中断による未確定バッファ破棄を検証
- [x] 送信中の `KeyboardInterrupt` でもセッションが送信成否不明のFAULT情報を保持することを検証
- [x] 模擬画像で実LeRobotDatasetの画像・動画保存と同期／非同期スレッド書込みをオフライン検証
- [x] 非同期書込みの完了を待ってから中断画像を削除し、遅延書込みと正常エピソードへの混入を防ぐことを検証
- [x] 保存前に全画像の読込みと寸法を検査し、統計サンプル対象外の欠損フレームも拒否することを検証
- [x] PNG書込み、Parquet保存、動画処理、中断処理、終了処理への例外注入でFAULTと保存境界を検証
- [ ] 複数プロセス書込み・複数カメラ・バッチ動画処理、ワーカー停止／ハングへの対応と検証
- [x] 起動時照合の失敗でセッション・ループをFAULTにし、送信・収録開始・保存を禁止することを検証
- [ ] 現物の校正・モーター設定照合と、実機運用での安全余裕幅の決定
- [ ] 急な指令変化・速度への制限／拒否処理とオフライン検証
- [ ] 通信異常・停止・トルク保持／解除・アーム落下防止の実機安全手順
- [ ] 安全確認後の監視下での限定的な実機テスト

**現状はエピソード収録処理までオフライン検証済みですが、実機での安全確認は未完了です。** 設定範囲内でも衝突、過負荷、急動作が起こる可能性があります。

### 3.5 AI認識とロボット動作の連携

- [x] 対象物：潰した空き缶、投入先：白い箱に決定
- [x] 使用AIモデル：OWLv2に決定、静止画像で認識動作を確認
- [ ] 認識結果に対応する動作の選択・指令生成
- [ ] 動作前の姿勢・経路・衝突確認
- [ ] OWLv2とFollowerの連携
- [ ] 自動回収／箱への投入と成功率測定

### 3.6 任意課題

- [x] テレオペレーションによる初期データセット収集
- [ ] 模倣学習モデルの学習・推論（任意）
- [ ] ROS 2連携（任意）

### 3.7 提出物

- [ ] GitHub公開用コード・READMEの最終整理
- [ ] 1〜3分のデモ動画
- [ ] 説明資料（PDF）
- [ ] YouTube動画のアップロード（少なくとも限定公開）
- [ ] omnicampusへのPDF・GitHub URL・動画URLの提出

## 4. テスト結果

2026年10月13日、WSL2のPython 3.12.3 / pytest 9.1.1で実行した結果：

```text
$ python -m pytest tests/ -v
collected 180 items
180 passed in 5.33s
```

収集180件、成功180件、失敗0件、スキップ0件。前STEPの153件に画像付き収録の統合テスト27件を追加しました。

| テストファイル | 成功件数 | 内容 |
| --- | ---: | --- |
| `tests/test_action_guard.py` | 18 | 関節範囲、不正値、欠損、余裕幅、入力不変性 |
| `tests/test_guarded_record_processor.py` | 4 | Processor互換性、F4違反の拒否、正常指令の受け渡し |
| `tests/test_safe_record_loop.py` | 15 | Fake Robot送信、戻り値の保存、FAULT、通信・書込・割り込み時の動作 |
| `tests/test_dataset_record_writer.py` | 27 | 観測・送信戻り値の対応、保存・破棄・終了、部分書込失敗、実LeRobotDataset読戻し |
| `tests/test_safe_record_session.py` | 25 | 状態遷移、保存禁止、割り込み、実データセットの複数エピソード保存・異常時破棄・保存不確実性 |
| `tests/test_record_startup_check.py` | 43 | ハッシュ・校正・単位・余裕幅照合、開始拒否とFAULT、読取り専用、LeRobot設定／校正型と実データセットの連携 |
| `tests/test_image_record_integration.py` | 27 | PNG／H.264読戻し、非同期中断、画像欠損・破損・寸法違反、部分保存・動画・終了失敗、ストリーミング拒否 |
| `tests/test_validate_detection.py` | 12 | ROI・信頼度、曖昧な認識結果、不正JSONの拒否 |
| `tests/test_validate_trajectory.py` | 9 | 設定範囲、データ整合性、指令変化、時間差診断、CLI |
| **合計** | **180** | **全件成功** |

実LeRobotDatasetの統合テストは `tmp_path` 配下に一時データセットを生成し、Fake RobotとNumPyの模擬画像で実施しました。実機・カメラ・ネットワークへの接続は行っていません。

- 正常系：2エピソード×3フレームを保存し、全6関節の `action` が `send_action()` の戻り値と一致することを確認。float32変換誤差を考慮（`rtol=1e-7`、`atol=1e-6`）し、actionと観測の異なる関節順序、観測値、時刻・各インデックス、保存メタデータも照合しました。
- 異常系：F2・F4の生指令／最終送信候補の範囲違反では送信を拒否。通信例外では送信成否不明のFAULTへ遷移し、セッションとループの再送信を禁止。明示的な中断後にParquetを読み、既に保存したエピソードのみが残ることを確認しました。
- 保存失敗：実際の保存後に例外を注入し、保存済みファイルが存在しても `commit_uncertain=True` を保持し、再保存・破棄・セッション終了を禁止することを確認しました。テスト終了時のネイティブファイルハンドルの解放は、セッションの回復やロールバックを意味しません。
- 起動時照合：模擬校正ファイルの変更・欠損・不正JSON、読込み済み校正の差異、`use_degrees`・Guard単位／範囲／余裕幅の不一致を拒否しました。ハッシュと範囲生成が同じ1回の読取り結果を使用し、合格時は開始だけでは送信しないこと、次のエピソードでも再照合することを確認。実LeRobotの設定・校正dataclassを持つFake Robotで、照合後に2フレームを実データセットへ保存・読戻しできました。
- 画像付き正常系：64×64のRGB模擬画像を2エピソード×3フレーム収録。同期書込み／2スレッドの非同期書込みを使用し、PNGを埋め込んだParquetとH.264動画を読戻して、全6関節のaction・観測、フレーム／エピソード番号、時刻、画像との対応を照合しました。PNGは画素の完全一致、動画は圧縮誤差として平均絶対誤差3未満・95パーセンタイル10以下（0〜255画素値）を確認しました。動画はPyAV 15.1.0、CPU H.264、`crf=18`、`preset=ultrafast`、`yuv420p`、30 fps設定で検証しました。
- 非同期中断：画像ワーカーをEventで一時停止し、中断処理が書込み完了を待つことを確認。手動中断・範囲違反・通信例外の各ケースで、未確定画像／バッファが破棄され、遅延書込みで画像が復活せず、以前の正常エピソードの画像／動画だけが残ることを確認しました。
- 画像エラー：PNG保存失敗に加え、破損・寸法違反を保存前に拒否。101フレーム中の統計サンプル対象外の1枚を失敗させると、LeRobot標準経路では100フレームの動画が正常確定する問題を再現しました。保存アダプターによる全画像検査の追加後は、ネイティブ保存前にFAULTとなり、Parquet／動画を確定せず明示的にバッファを破棄できます。
- 部分保存・終了エラー：実Parquet書込み後、動画処理中、ネイティブ終了後に例外を注入し、不確実な状態を保持して再保存・破棄・終了を禁止することを確認。部分ファイルは自動削除せず、隔離・手動確認の対象です。画像削除後の例外も正常な中断完了とは報告しません。

上記の成功は、テストで定義した条件に対する結果です。**実機の衝突安全性や自動回収の成功を証明するものではありません。** 起動時照合のテストは一時ファイルと模擬校正値を使用し、セッションの実データセット統合テストはリポジトリ内のFollowerスナップショット `my_follower_for_004.json` を読取り専用で使用しています。起動時照合のロジックは検証済みですが、現物の校正状態・モーターレジスタとの一致は未検証です。時刻はLeRobotがFPS設定から生成する値であり、実時間での30 Hz収録性能は未検証です。

実行方法：

```bash
cd ~/projects/so101
source .venv/bin/activate
python -m pytest tests/ -v
```

## 5. ソフトウェア構成

```text
src/
├── control/
│   ├── action_guard.py
│   ├── guarded_record_processor.py
│   ├── safe_record_loop.py
│   ├── dataset_record_writer.py
│   ├── safe_record_session.py
│   ├── record_startup_check.py
│   └── validate_trajectory.py
└── vision/
    ├── detect_objects.py
    └── validate_detection.py

tests/
├── test_action_guard.py
├── test_guarded_record_processor.py
├── test_safe_record_loop.py
├── test_dataset_record_writer.py
├── test_safe_record_session.py
├── test_record_startup_check.py
├── test_image_record_integration.py
├── test_validate_detection.py
└── test_validate_trajectory.py
```

### エピソード内の制御フロー（オフラインテスト対象）

```text
SafeRecordSession.start()：指定されたRecordStartupCheckで照合 → 論理的な収録開始
  ↓
DatasetRecordWriter.stage_observation()：取得済み観測をコピー・検証
  ↓
Leader由来の指令（テストでは模擬指令）
  ↓
Action Guard：生指令の検証
  ↓
teleop_action_processor → Action Guard再検証
  ↓
robot_action_processor → Action Guard再検証
  ↓
robot.send_action()（Fake Robotテスト済み）
  ↓
戻り値をAction Guardで検証
  ↓
DatasetRecordWriterへ戻り値の「送信目標指令」を渡す
  ↓
LeRobotDataset.add_frame()：観測と送信目標指令を未確定バッファへ追加
  ↓
全フレーム成功 → SafeRecordSession.save_episode()
  → 非同期画像書込みの完了待ち → 全画像検査 → ネイティブsave_episode()
  → 明示的なfinalize()
異常／中断 → FAULT／明示的なabort_episode()（ネイティブ保存開始後の失敗時は破棄禁止）
```

`SOFollower.send_action()` は設定により `max_relative_target` で指令を制限し、その送信目標指令を返します。標準のLeRobot 0.6.1 `record_loop()` は送信戻り値 `_sent_action` ではなく送信前の `action_values` をデータセット用に使用する実装が確認されたため、収録時の整合性には注意が必要です。

`safe_record_loop.py` は1フレームの指令検証と1回の送信を担当します。生指令、各Processor出力、最終送信候補を検証し、送信戻り値を保存コールバックへ渡します。戻り値の検証は送信後のデータ整合性検査です。`FAULT` は新規指令を禁止しますが、既に送信された動作の物理停止やトルク制御は行いません。

`dataset_record_writer.py` は観測を送信前に検証・コピーし、送信戻り値をfeature名の順序でfloat32の `action` に変換して `add_frame()` に渡します。エピソード確定、未確定バッファ破棄、終了は明示的な操作です。ネイティブ保存の開始前に、非同期画像書込みを待って全RGB PNGをデコードし、フレーム数・形式・寸法を確認します。欠損・破損・寸法違反はFAULTとなり、ネイティブ保存を呼ばないため未確定バッファの明示的な破棄が可能です。ネイティブ保存開始後の失敗と終了失敗は `commit_uncertain` を保持し、ロールバック成功を仮定しません。

全画像検査はLeRobot 0.6.1の `dataset.writer.episode_buffer` と `image_writer.wait_until_done()` に依存します。メモリ内のFakeDatasetには検査対象のファイルがありません。ストリーミング動画エンコードはファイルによる検査ができないため、writer構築時に拒否します。全画素の検査にはI/O負荷があり、実カメラ解像度・実時間での性能や、画像ワーカーが停止／ハングした場合の待機上限は未検証です。

`safe_record_session.py` は両者を1エピソード単位で統括します。正常時は `READY → RECORDING → SAVED → FINALIZED`、収録異常時は `FAULT → 明示的なABORTED → FINALIZED`、手動中断時は `RECORDING → ABORTED → FINALIZED` です。ネイティブ保存開始後の失敗時は `FAULT` と不確実性を保持し、再送・再保存・破棄・終了を禁止します。隔離して手動確認することは運用上の要件であり、自動隔離機能は未実装です。次の正常エピソードには同じwriterと新しいloop/sessionを使い、最後に一度終了します。構成要素を独立に操作したり、並行使用したりしないでください。

`record_startup_check.py` の `RecordStartupCheck` は校正ファイルのパス、事前確認済みの固定SHA-256、明示的な `margin_deg` / `margin_gripper` を受け取ります。`SafeRecordSession(loop, writer, startup_check=check)` として渡すと、`start()` 内で、同じファイル内容のハッシュ・6関節の校正形式・送信先の読込み済み校正・`config.use_degrees is True`・Guardの範囲／単位／余裕幅を照合します。失敗時は未送信のFAULTとなり、開始や送信を再試行できません。合格時は `startup_result` にハッシュと範囲・余裕幅を保持します。SHA-256はレビュー済みFollowerスナップショットから事前に固定し、起動時に現在ファイルから自動採用しないでください。

互換性のため `startup_check` を省略した従来のオフラインAPIも使用できますが、その場合 `startup_result` は `None` で、照合済みとは扱いません。各エピソードの開始時に照合を指定し、開始後は校正・設定・Guardを変更しない運用が必要です。この照合はローカル設定の一致を確認するもので、実機の識別・接続・モーターレジスタやバスの設定確認は行いません。テストの余裕幅は模擬値であり、実機で使う数値は未決定です。

急動作対策は、初回の最終送信候補と取得済み関節観測との差、以後の候補と直前の送信目標との差、単調時計による経過時間を送信前に検査して、違反時は補正せず拒否する方針です。閾値、観測の鮮度、通信遅延・長い時間差の扱い、関節ごとの単位を決めてからオフライン実装・検証します。現在の起動時照合や範囲内判定には、この変化量・速度の検査は含まれません。

## 6. キャリブレーションと過去の収録データ

### キャリブレーション

現在のキャリブレーション保存先：

- Follower：`~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
- Leader：`~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json`

004収録に向けたスナップショット：

- `configs/calibration_snapshots/my_follower_for_004.json`
- `configs/calibration_snapshots/my_leader_for_004.json`

両スナップショットは、それぞれ対応するキャリブレーションJSONとのSHA-256一致を事前に確認済みです（収録直前にも再確認します）。LeaderとFollowerではF2・F4の校正設定範囲に差があるため、**Follower側の設定範囲を用いてモーター指令を検証**します。

### 過去のデータセット

| データセット | 既知の状態 | 運用判断 |
| --- | --- | --- |
| `so101_teleop_001` | 初期テレオペ収録、1エピソード898フレーム、カメラなし | 過去の動作確認用 |
| `so101_pick_place_002` | F4設定範囲違反1,225フレーム | 自動再生に使用しない |
| `so101_pick_place_003` | F4設定範囲違反350フレーム | 自動再生に使用しない |
| `so101_pick_place_004` | **未収録** | 安全確認後に新規収録予定 |

`validate_trajectory.py` による検証はオフライン診断です。範囲・連続性の結果が良好でも、衝突安全性や自動再生の可否は保証しません。

## 7. LeRobotDatasetのAPI・オフライン統合確認（2026-10-13）

インストール済みLeRobot 0.6.1のAPIと保存処理を確認し、以下を実データセットでオフライン検証しました。LeRobotライブラリ本体は変更していません。

| メソッド | 想定する役割 | 状態 |
| --- | --- | --- |
| `LeRobotDataset.create(...)` | 新しい収録データセットの生成 | `tmp_path` 配下で生成成功 |
| `dataset.add_frame(frame)` | 1フレームを収録バッファへ追加 | 観測・送信戻り値の複数フレーム追加成功 |
| `dataset.save_episode(...)` | エピソードの保存 | 画像なし／模擬RGB画像／H.264動画で複数エピソード保存・読戻し成功 |
| `dataset.clear_episode_buffer(...)` | エピソードバッファのクリア | 非同期画像の書込み完了待ち、異常・中断画像の破棄、既存正常エピソードの保持を確認 |
| `dataset.finalize()` | データセット処理の終了 | 正常保存後・中断後の明示的な終了成功 |

検証範囲は関節指令・観測、模擬RGB画像のPNG／H.264保存・読戻し、同期／スレッドによる非同期書込み、エピソードライフサイクルと例外注入です。複数プロセス、複数カメラ、バッチ動画処理、ワーカー停止／ハング、実カメラ映像の取得・同期、実機制御は未検証です。ネイティブ保存開始後の失敗は完全なロールバックとみなさず、不確実状態のデータセットを正常データとして扱いません。部分ファイルを残し、確認済みの正常エピソードと区別して手動で調査する方針です。自動隔離、事故復旧・データセット再利用の手順は未実装・未検証です。

## 8. 実施済みの動作確認・操作メモ

### カメラ

- V4L2で `/dev/video0` の認識を確認。
- 640×480 / 30 fpsなどの対応モードを確認。
- Python/OpenCVで画像取得・保存成功。
- 実カメラを使ったLeRobot収録は未検証（模擬画像による保存・読戻しは検証済み）。

### 初回LeRobotDataset（2026-10-11）

- 保存先：`~/projects/so101/datasets/so101_teleop_001`
- エピソード数：1
- フレーム数：898
- FPS設定：30（収録時に26.5 Hzの警告が一度発生）
- 主な記録項目：`action`、`observation.state`、`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`
- カメラ画像なし、Hugging Face Hub未アップロード

### 実行前の確認に使うコマンド（これだけではモーターを動かさない）

```bash
cd ~/projects/so101
source .venv/bin/activate
ls -l /dev/serial/by-id/
v4l2-ctl --list-devices
python -m pytest tests/ -v
```

`/dev/ttyACM0` / `/dev/ttyACM1` は再接続で入れ替わる場合があります。実機を使用するコマンドは、ポートを確認し、アームを支持し、停止方法を定めたうえで別途実行します。**現段階では004収録用の実機コマンドは提供しません。**

## 9. 次に行うこと（優先順）

起動時のローカル校正・設定照合と、模擬画像を使う画像付き収録のオフライン検証は完了しました。次のSTEPは急動作対策のオフライン実装・検証です。

1. **急動作対策をオフライン実装／検証**：初回観測と候補指令、直前の送信目標と次の候補指令を用いて、関節ごとの変化量・速度・観測鮮度・時間差を検査する。実機での安全余裕幅・閾値は別途決定する。
2. **画像書込みの停止・隔離手順を整備**：ワーカー停止／ハング時の待機上限、保存不確実なデータセットの隔離と手動調査・復旧手順を設計／検証する。
3. **実機の停止・落下対策を確認**：特に `disconnect()` が `disable_torque_on_disconnect` 設定に従ってトルク解除を要求する点を確認する。ソフトウェア停止のみを安全策としない。
4. **監視下で限定的な実機テストと004の新規収録**：現物と現在のFollower校正・モーター設定を照合し、旧002/003を自動再生しない。
5. **004をオフライン検証**：設定範囲、時系列、指令変化量と記録内容を確認する。
6. **OWLv2と安全に検証した動作の連携**：認識結果を動作選択へ接続し、空き缶回収・箱投入を実機評価する。
7. **提出物の完成**：README、結果、デモ動画、PDFを整理する。

## 10. 安全上の注意

- **Action Guardの合格は実機動作の許可ではありません。** 校正由来の設定範囲のみを検査しており、衝突、障害物、モーター負荷、把持物の落下までは評価していません。
- LeRobot標準の`record()`は終了時に `robot.disconnect()` を呼びます。Followerの設定によっては、その際のトルク解除でアームが落下する可能性があります。安全に支持し、実際の終了手順を確認してから実機試験をします。
- 送信時の例外は、指令がモーターバスへ到達したか不明な状態です。安易な再送信や自動再開を行いません。
- ソフトウェアの `FAULT` は物理的な非常停止ではありません。収録コンポーネントはトルク解除や切断を自動実行せず、落下防止と物理停止は別途確認します。
- ネイティブ保存開始後の失敗では部分的なファイルやメタデータが残る可能性があります。`commit_uncertain` を保持して隔離・手動確認し、自動再保存やバッファ破棄で正常状態へ戻しません。開始前の画像検査失敗は未確定バッファを明示的に破棄できます。
- 通電中にサーボ用3ピンケーブルを抜き差ししません。
- `Input voltage error!` などの電源関連エラーが発生した場合は、動作を繰り返さず電源・接続を点検します。
- ローカルデータセット、機器識別情報、認証情報、個人情報を意図せず公開リポジトリに含めません。

## 11. AIモデル

- **Model:** Google OWLv2
- **Model ID:** [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble)
- **Task:** Zero-Shot Object Detection
- **License:** Apache License 2.0（採用メモ。公開・配布時にモデル配布元の表示を再確認する）

現在の認識は静止画像に対するオフライン検証です。ロボットの把持位置推定や動作計画まで実現しているわけではありません。

## 12. 最終課題提出条件

- **提出期限：2026-11-02（月）10:00（日本時間）**
- **GitHub：** 公開リポジトリURL（ソースコード一式）
- **動画：** 1〜3分、YouTubeで少なくとも限定公開（倍速使用時は明記）
- **説明資料：** PDF（目的、システム構成、AIの利用方法、工夫、実験結果、考察）
- **提出先：** omnicampus（PDF、GitHub URL、YouTube URL）

公開・提出前に、氏名・所属・メールアドレスなどの個人情報と秘密情報の混入を確認します。
