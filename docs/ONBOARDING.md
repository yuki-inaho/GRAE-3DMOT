# LLMオンボーディングサマリー

> 新任LLMエージェントがこのリポジトリ（GRAE-3DMOT / self-contained uv + CUDA 12.8 版）に参加する際の初期資料です。
> 作業を始める前に必ず一読してください。記入済みドキュメントとしてバージョン管理しています（更新時はこのファイルも更新）。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** GRAE-3DMOT — Geometry Relation-Aware Encoder for Online 3D Multi-Object Tracking（nuScenes 上の3次元多物体追跡 / CVPR 2025 のbeta実装）。
- **最終成果物:** 元の研究リポジトリ（conda + PyTorch 1.9 + 外部CUDA拡張前提）を、**外部依存を同梱せず（self-contained）**・**uv管理**・**CUDA 12.8 のGPUで動く**形に移行した実行環境一式と、それを保証するテスト/型/フォーマットの仕組み。
- **ビジネス背景・価値:** 開発機のGPUが **NVIDIA RTX PRO 4000 Blackwell（compute capability sm_120）** で、元構成（`torch==1.9.0+cu111`）では物理的に起動できない。再現・改良のため、submodule/外部cloneに頼らず誰でも `uv sync` 一発で動かせる状態にすることが価値。
- **現時点の進捗サマリ:** 移行完了。`uv run pytest` 50件pass、`uv run ty check` クリーン、GPU上でモデルの forward/backward と前処理opsを実機検証済み。デフォルトブランチは `cu128`。**唯一未実施なのは実nuScenesデータ + CenterPoint検出JSONを使った全量の前処理〜学習の通し実行**（データが手元に無いため。コードパスは合成データで検証済み）。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ラインです。
- **CUDA 12.8 / Blackwell 前提を崩さない。** `pyproject.toml` の PyTorch は `pytorch-cu128` インデックス由来（`torch>=2.7,<2.9`）。`torch.version.cuda == "12.8"` を維持（`tests/test_device_and_env.py::test_torch_built_with_cuda_12_8` がガード）。cu111 / torch1.9 へ戻さない。
- **self-contained を維持する。** `iou3d_nms_cuda`（CenterPoint）と SimpleTrack の `nms`/`nu_array2mot_bbox` は `ops/` 配下の純Python/PyTorch実装で完結。`import iou3d_nms_cuda` / `from SimpleTrack ...` / `from mot_3d ...` / `mmdet3d` を**復活させない**（submodule・`pip install -e` 外部cloneも禁止）。
- **BEV IoU の軸規約は CenterPoint 準拠を保つ。** box は `[x, y, z, dx, dy, dz, heading]`、col[3]=dx が local-x 方向の幅。`tests/test_regression.py::test_bev_iou_axis_convention_golden` と `tests/test_iou3d_bev.py::test_bev_iou_centerpoint_axis_convention` がgolden値で固定。SimpleTrack側NMSのIoUは別途その幾何規約に一致させている（混同しない）。
- **テストとゴールデン値を緑に保つ。** `uv run pytest`（50件）と `uv run ty check` は常に pass。回帰テストの golden 値（IoU=1/3, 1/9, 45°≈0.7071, NMS keep=[0,3,2] 等）を勝手に書き換えない。
- **ランタイム型契約を尊重する。** `ops/` と `utils/device.py` は jaxtyping + beartype で形状/型を実行時強制。注釈と実際のテンソル形状が食い違うとテストが落ちる。
- **エントリポイントと公開API互換を壊さない。** `single_gpu_train.py --device`、`tools/convert_dataset.py`、`validation.py`、`boxes_iou_bev_cpu`（in-place/return両対応）など既存の呼び出し方を維持。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| セットアップ/実行手順 | [`README.md`](../README.md) | uvセットアップ、cu128方針、self-containedな前処理ops、実行コマンド、テスト/型チェックの一次資料 |
| 依存・ツール定義 | [`pyproject.toml`](../pyproject.toml) | 依存（cu128 index, jaxtyping/beartype, dev: pytest/ty/ruff）、`[tool.uv]`/`[tool.ty]`/`[tool.ruff]` 設定 |
| ロックファイル | [`uv.lock`](../uv.lock) | linux/x86_64 に固定した決定的な依存解決結果 |
| 自前ops（要求の中核） | [`ops/iou3d_nms_cuda.py`](../ops/iou3d_nms_cuda.py), [`ops/simpletrack_nms.py`](../ops/simpletrack_nms.py) | CenterPoint BEV IoU / SimpleTrack NMS の self-contained 実装 |
| テスト資産 | [`tests/`](../tests/) | 単体・回帰・型強制・GPUテスト（`test_iou3d_bev` / `test_simpletrack_nms` / `test_device_and_env` / `test_model_gpu` / `test_regression` / `test_typing`） |
| 環境診断 | [`scripts/check_environment.py`](../scripts/check_environment.py) | 依存import・CUDA12.8・GPU・self-contained opsのスモーク確認 |
| オンボーディング | [`docs/ONBOARDING.md`](ONBOARDING.md) | 本ファイル。新任エージェント向けの要求・制約・運用ルールの集約 |
| 既知課題 | 本ファイル §2・§4 と README 内の注記 | 専用の課題管理表は未整備（TBD）。重要事項は本ドキュメントに集約 |

> 注: フォーマルな「要求定義書 / 要件定義書 / WBS」の独立文書は未整備。要求・制約は §1・§2 に、進捗は §1 末尾に集約している。

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- `ops/` の self-contained 実装、`utils/`・`tests/`・`scripts/` の改良とテスト追加。
- 依存・ツール設定（`pyproject.toml` の cu128/dev設定、`uv.lock` 更新、ruff/ty設定）。
- README やこの ONBOARDING など docs の更新。
- 既存エントリポイントのバグ修正・型強化・回帰テスト追加（互換維持の範囲で）。

### 任せないタスク（事前確認が必要）
- CUDA/PyTorch のメジャー方針変更（cu128→他、torchダウングレード等）や Python バージョン変更。
- self-contained 方針に反する外部依存の再導入（CenterPoint/SimpleTrack/mmdet3d の submodule・外部clone・`pip install -e`）。
- BEV IoU / NMS の数値規約変更や golden 値の改変（アルゴリズムを変える提案は要合意）。
- `git push --force`、デフォルトブランチ変更、`main`/`cu128` の削除など、リポジトリ運用に影響する破壊的操作。
- 実nuScenesデータ・公開検出結果・checkpoint の同梱や再配布。

## 5. インタラクション方針
- **回答スタイル:** 日本語、結論先出し。見出し＋箇条書き、コマンドやファイルパスはコードスパンで明示（クリックできる相対パスを優先）。
- **回答手順:** 現状確認 → 変更方針（必要なら選択肢と推奨）→ 実施 → 実機での検証結果を提示、の順。
- **禁止事項・注意:** 未検証の内容を「動く」と断定しない（必ず `pytest`/`ty`/スモークで裏付ける）。スタブ・仮置き実装で「完了」としない。破壊的・外部公開操作は明示の指示がある時のみ。
- **秘匿情報の扱い:** トークン/認証情報をログや成果物に残さない。nuScenes等のデータや checkpoint はコミットしない（`.gitignore` と `temp/` を活用）。

## 6. 試行タスク（オンボーディング演習）
1. `uv sync` 後に `uv run python scripts/check_environment.py` を実行し、`CUDA build 12.8` / `Blackwell` / 自前opsOK を確認する。
2. `uv run pytest -q` と `uv run ty check` を実行し、50件pass・型チェッククリーンを確認する。失敗時はどのテストがどの契約を守っているか説明する。
3. `ops/iou3d_nms_cuda.py` の `boxes_iou_bev` に対し、誤った形状のテンソルを渡すと `TypeCheckError` が出ることを再現し、jaxtyping+beartype の役割を一段落で説明する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** コード変更で前提が変わったら README と本 ONBOARDING を同時更新。事実はコード/テストで裏取りしてから書く。
- **TBDの扱い:** 未確定は「TBD」と明示し、断定しない。専用の要求/要件/WBS文書を新設する場合は §3 の表に追記する。
- **レビュー/承認フロー:** 作業は `cu128`（デフォルトブランチ）へ。コミットは小さく意味単位で、メッセージ末尾に `Co-Authored-By` を付与。push/コミットはユーザー依頼時に実施。マージ/PR・デフォルトブランチ変更は明示指示時のみ。
- **その他の運用ルール:** コミット前に `uv run pytest` / `uv run ty check` / `uv run ruff format --check` を通す。`temp/` はスクラッチ用で gitignore 済み（成果物を置かない）。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `yuki-inaho/GRAE-3DMOT`（default: `cu128`）。`ops/`（自前ops）, `models/`（GRAEモデル）, `trainer/`, `dataset/`, `tools/convert_dataset.py`（前処理）, `tests/`, `scripts/`。
- **代表的なコマンド:**
  ```shell
  uv sync                         # 環境構築（Python3.10 + torch cu128）
  uv sync --extra nuscenes        # nuScenesデータ/評価を使う場合のみ
  uv run python scripts/check_environment.py
  uv run pytest                   # 単体+回帰テスト（GPUテストはCUDA無しなら自動skip）
  uv run ty check                 # 静的型チェック（型付きモジュール限定）
  uv run ruff format              # フォーマット
  uv run python single_gpu_train.py --device cuda:0   # 単一GPU学習（要データ）
  ```
- **依存ライブラリ:** torch 2.8.0+cu128 / torchvision 0.23.0+cu128、numpy(<2)/scipy/shapely/pyquaternion、jaxtyping+beartype（実行時型検査）、fvcore/lapx/pandas/matplotlib/tensorboard、（任意）nuscenes-devkit/motmetrics、dev: pytest/ty/ruff。
- **環境前提:** Linux x86_64、NVIDIA Blackwell（sm_120）等 CUDA 12.8 対応GPU。`nvcc` 不要（cu128 wheel同梱ランタイムを使用）。
- **連絡先/責任者:** リポジトリオーナー（yuki-inaho）。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理してください。
