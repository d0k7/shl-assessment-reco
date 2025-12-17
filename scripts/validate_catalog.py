import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/catalog.jsonl")
    args = ap.parse_args()

    count = 0
    uniq = set()
    missing_name = 0

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            count += 1
            uniq.add(obj.get("url"))
            if not (obj.get("name") or "").strip():
                missing_name += 1

    print("Total items:", count)
    print("Unique URLs:", len(uniq))
    print("Missing name:", missing_name)
    if count < 377:
        print("WARNING: Below 377. Increase sitemap coverage or add better seeds.")
    else:
        print("OK: Meets >= 377 requirement.")


if __name__ == "__main__":
    main()
