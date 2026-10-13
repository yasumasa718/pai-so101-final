"""問題のあるデータセットに「使用禁止」の印を付ける。ファイルは移動・削除しない。

異常が起きても、画像を保存する裏側の処理が動き続けることがある。
そのため、印となるファイルを作り、この収録処理でのデータセット全体の再利用を禁止する。
ファイルのアクセス権を変更する仕組みではない。他の読み込み処理でも
assert_dataset_not_quarantined() を呼んで、使用禁止の印を確認する必要がある。
記録には、例外の詳しいメッセージ、機器の識別番号、認証情報、絶対パスを含めない。
印の保存に失敗しても、メモリ内の FAULT（異常を記憶した状態）は解除しない。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

QUARANTINE_FILENAME = "SO101_QUARANTINE.json"
_excluded_roots: set[Path] = set()


class DatasetQuarantinedError(RuntimeError):
    """人による確認が済んでおらず、データセットの使用が禁止されていることを表す例外。"""


def assert_dataset_not_quarantined(root: Path) -> None:
    """使用禁止の印があれば例外を発生させる。印がないことだけではデータの正常性は保証しない。"""
    marker = root / QUARANTINE_FILENAME
    # 印が壊れていたりリンク先がなくても使用を拒否する。リンク先の内容は読まない。
    if root.resolve() in _excluded_roots or marker.exists() or marker.is_symlink():
        raise DatasetQuarantinedError("Dataset has a quarantine marker; manual review required")


def record_dataset_quarantine(
    root: Path,
    *,
    stage: str,
    error_type: str,
    commit_uncertain: bool,
    cleanup_uncertain: bool,
    frames_in_episode: int,
    saved_episodes: int,
) -> Path:
    """使用禁止の印を新しく保存する。すでにある印は上書きしない。

    書き込みが途中で失敗した不完全な印でも、再利用は禁止する。
    保存先は既存のデータセットに限る。異常処理の中で新規作成や移動はしない。
    """
    # 印をディスクに書けなくても、このプログラムの実行中は再利用を禁止する。
    # 再起動後も禁止を続けるには、人が別途記録を残すなどの対応が必要。
    _excluded_roots.add(root.resolve())
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Quarantine requires an existing local dataset directory")
    marker = root / QUARANTINE_FILENAME
    payload = {
        "schema_version": 1,
        "status": "quarantined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_stage": stage,
        "error_type": error_type,
        "commit_uncertain": commit_uncertain,
        "cleanup_uncertain": cleanup_uncertain,
        "frames_in_episode": frames_in_episode,
        "saved_episodes": saved_episodes,
        "rollback_confirmed": False,
        "workers_confirmed_stopped": False,
        "manual_review_required": True,
    }
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        return marker
    return marker
