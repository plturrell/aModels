#!/usr/bin/env python3
"""Download and reconstruct model assets from a GitHub Release."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError("Attempted Path Traversal in Tar File")
    tar.extractall(path=path)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open('wb') as handle:
        shutil.copyfileobj(response, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag containing the assets")
    parser.add_argument("--repo", default="plturrell/aModels", help="GitHub repo in owner/name format")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Destination directory (defaults to current working directory)",
    )
    args = parser.parse_args()

    base_url = f"https://github.com/{args.repo}/releases/download/{args.tag}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    manifest_url = f"{base_url}/manifest.json"
    try:
        download_file(manifest_url, manifest_path)
    except urllib.error.HTTPError as exc:  # pragma: no cover - external dependency
        raise SystemExit(f"Failed to download manifest ({manifest_url}): {exc}")

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for asset in manifest.get("assets", []):
        parts = []
        for part_name in asset["parts"]:
            part_url = f"{base_url}/{part_name}"
            part_path = output_dir / part_name
            print(f"⬇️  Downloading {part_url}")
            download_file(part_url, part_path)
            parts.append(part_path)

        target_path = output_dir / asset["target_path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if asset["type"] == "file":
            with target_path.open('wb') as outfile:
                for part_path in parts:
                    with part_path.open('rb') as infile:
                        shutil.copyfileobj(infile, outfile)
            print(f"✅ Restored {target_path}")
        else:
            archive_name = asset.get("archive")
            if not archive_name:
                raise RuntimeError(f"Archive name missing for asset {asset['name']}")
            archive_path = output_dir / archive_name
            with archive_path.open('wb') as archive_handle:
                for part_path in parts:
                    with part_path.open('rb') as infile:
                        shutil.copyfileobj(infile, archive_handle)
            with tarfile.open(archive_path, 'r:gz') as tar:
                safe_extract(tar, target_path.parent)
            archive_path.unlink(missing_ok=True)
            print(f"✅ Extracted {target_path}")

        for part_path in parts:
            part_path.unlink(missing_ok=True)

    manifest_path.unlink(missing_ok=True)
    print("All assets restored. Consider verifying checksums before use.")


if __name__ == "__main__":
    main()
