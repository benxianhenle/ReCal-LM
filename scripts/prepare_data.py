import argparse
import json
import re
import urllib.request
from pathlib import Path


def load_conf(path: str | Path) -> dict:
    conf = {}
    pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$')
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            conf[match.group(1)] = match.group(2)
    return conf


def main():
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
            data = row.get("row", {})
            text = data.get("text")
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
    print(f"saved={out} rows={written}")


if __name__ == "__main__":
    main()

