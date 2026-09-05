"""Rebuild index.html from an existing manifest.json without recomputing embeddings."""
import json
import sys
from pathlib import Path

import visualize_bacpipe as vb


def main() -> int:
    for arg in sys.argv[1:]:
        d = Path(arg)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        (d / "index.html").write_text(vb._index_html(manifest), encoding="utf-8")
        n = len(manifest["models"])
        print("%s/index.html  <- %d models" % (d, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
