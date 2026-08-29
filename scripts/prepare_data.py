"""Fetch a small FineWeb-Edu rows-API sample and write it as JSONL text.

中文：拉取一小份 FineWeb-Edu rows API 样本，并写成 JSONL 文本。"""

import argparse
import json
import re
import urllib.request
from pathlib import Path


def load_conf(path: str | Path) -> dict:
    """Read simple KEY=\"VALUE\" entries from a local config file.

    中文：从本地配置文件读取简单的 KEY=\"VALUE\" 项。"""

    conf = {}
    pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$')
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            conf[match.group(1)] = match.group(2)
    return conf


def main():
    """Download rows from the configured API endpoint into a text JSONL file.

中文：从配置的 API 端点下载 rows，并写入文本 JSONL 文件。"""

    parser = argparse.ArgumentParser(description="Fetch a tiny rows-API sample into JSONL.")
    parser.add_argument("--conf", default=".conf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    conf = load_conf(args.conf)
    url = conf.get("fineweb-edu_API")
    if not url:
        raise ValueError("fineweb-edu_API was not found in the conf file")

    request = urllib.request.Request(url)
    token = conf.get("HuggingFaceFW_Access_Token")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("rows", [])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            # The rows API nests dataset columns under the "row" key.
            data = row.get("row", {})
            text = data.get("text")
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
    print(f"saved={out} rows={written}")


if __name__ == "__main__":
    main()
