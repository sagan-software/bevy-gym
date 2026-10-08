"""Keep a page, its workers, and its models in one content-addressed directory.

Trunk 0.21.14 leaves worker and copy-file names unversioned:
https://github.com/trunk-rs/trunk/blob/v0.21.14/guide/src/assets/index.md
Its post_build hook runs on unpublished staging output:
https://github.com/trunk-rs/trunk/blob/v0.21.14/guide/src/build/hooks.md
"""

import hashlib
import os
from pathlib import Path


def version_assets(root: Path) -> str:
    """Move related assets together before publishing their importing page."""
    entrypoint = root / "index.html"
    html = entrypoint.read_text(encoding="utf-8")
    marker = 'src="app.js"'
    if html.count(marker) != 1:
        raise ValueError("Expected exactly one application entrypoint")
    assets = sorted(
        path for path in root.iterdir()
        if path.name != "index.html" and path.suffix != ".css"
    )
    digest = hashlib.sha256()
    # Hash names and contents in stable order; stream large WebAssembly files.
    for asset in assets:
        files = sorted(asset.rglob("*")) if asset.is_dir() else [asset]
        for path in files:
            if path.is_file():
                name = path.relative_to(root).as_posix().encode("utf-8")
                digest.update(len(name).to_bytes(8, "big"))
                digest.update(name)
                with path.open("rb") as source:
                    content_hash = hashlib.file_digest(source, "sha256")
                    digest.update(content_hash.digest())
    directory = f"build-{digest.hexdigest()}"
    destination = root / directory
    destination.mkdir()
    for asset in assets:
        asset.rename(destination / asset.name)
    # CSS already has Trunk's content hash and can stay at the public root.
    entrypoint.write_text(
        html.replace(marker, f'src="{directory}/app.js"'), encoding="utf-8"
    )
    return directory


def main() -> None:
    """Read Trunk's staging directory at the process boundary."""
    version_assets(Path(os.environ["TRUNK_STAGING_DIR"]))


if __name__ == "__main__":
    main()
