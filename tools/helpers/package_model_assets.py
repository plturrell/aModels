#!/usr/bin/env python3
"""Package large model assets into split archives for GitHub Releases."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

CHUNK_DIGITS = 3


def split_file(src: Path, dest_prefix: str, out_dir: Path, chunk_size: int) -> list[str]:
    parts: list[str] = []
    with src.open('rb') as infile:
        idx = 0
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            part_name = f"{dest_prefix}{idx:0{CHUNK_DIGITS}d}"
            part_path = out_dir / part_name
            with part_path.open('wb') as handle:
                handle.write(chunk)
            parts.append(part_name)
            idx += 1
    return parts


def make_archive(source_dir: Path, name: str) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_base = Path(tmpdir) / name
        archive_path_str = shutil.make_archive(
            base_name=str(tmp_base),
            format='gztar',
            root_dir=str(source_dir.parent),
            base_dir=source_dir.name,
        )
        archive_path = Path(archive_path_str)
        target = source_dir.parent / f"{name}.tar.gz"
        shutil.move(str(archive_path), target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="scripts/model_assets_manifest.json",
        help="Path to the asset manifest JSON (relative to repo root).",
    )
    parser.add_argument(
        "--source-root",
        help="Directory containing the original model assets (defaults to agenticAiETH_layer4_Models if present, otherwise models).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/model-release",
        help="Where to write the packaged release artifacts.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        help="Override chunk size in MB (default taken from manifest).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_cfg = json.loads(manifest_path.read_text(encoding='utf-8'))
    chunk_size_mb = args.chunk_size_mb or manifest_cfg.get("chunk_size_mb", 1500)
    chunk_size = chunk_size_mb * 1024 * 1024

    if args.source_root:
        source_root = Path(args.source_root).expanduser().resolve()
    else:
        candidate = repo_root / "agenticAiETH_layer4_Models"
        source_root = candidate if candidate.exists() else (repo_root / "models")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    release_manifest = {
        "chunk_size_mb": chunk_size_mb,
        "generated_from": str(source_root),
        "assets": [],
    }

    for asset in manifest_cfg.get("assets", []):
        rel_path = Path(asset["relative_path"])
        source_path = source_root / rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing asset: {source_path}")
        name = asset["name"]
        if asset["type"] == "file":
            parts = split_file(
                source_path,
                dest_prefix=f"{name}.part",
                out_dir=output_dir,
                chunk_size=chunk_size,
            )
            release_manifest["assets"].append(
                {
                    "name": name,
                    "type": "file",
                    "source": str(source_path),
                    "target_path": f"models/{asset['relative_path']}",
                    "parts": parts,
                }
            )
        elif asset["type"] == "directory":
            archive_path = make_archive(source_path, name)
            try:
                parts = split_file(
                    archive_path,
                    dest_prefix=f"{name}.tar.gz.part",
                    out_dir=output_dir,
                    chunk_size=chunk_size,
                )
            finally:
                archive_path.unlink(missing_ok=True)
            release_manifest["assets"].append(
                {
                    "name": name,
                    "type": "directory",
                    "source": str(source_path),
                    "target_path": f"models/{asset['relative_path']}",
                    "archive": f"{name}.tar.gz",
                    "parts": parts,
                }
            )
        else:
            raise ValueError(f"Unknown asset type: {asset['type']}")

    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(release_manifest, indent=2), encoding='utf-8')
    print(f"✅ Packaged assets written to {output_dir}")
    print("   Upload manifest.json and all part files to a GitHub release.")


if __name__ == "__main__":
    main()
