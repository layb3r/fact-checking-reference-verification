import os
import json
import sys
import uuid
import argparse
import time


def load_env(env_path: str):
    if not os.path.exists(env_path):
        print(f"Warning: {env_path} not found, skipping .env load")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            os.environ.setdefault(key, val)


load_env(os.path.join(os.path.dirname(__file__), "retriever", ".env"))

_retriever_dir = os.path.join(os.path.dirname(__file__), "retriever")
if _retriever_dir not in sys.path:
    sys.path.insert(0, _retriever_dir)

from retriever.submodules.reference_schema import ExtractedCitation, Identifiers
from retriever.main_retriever import retrieve_paper


def instance_to_citation(instance: dict, idx: int) -> ExtractedCitation:
    meta = instance["citation_metadata"]
    ids = meta.get("identifiers", {}) or {}
    ref_id = uuid.uuid4().hex[:12]
    return ExtractedCitation(
        ref_id=ref_id,
        raw_text=None,
        title=meta.get("title"),
        authors=meta.get("authors", []) or [],
        venue=meta.get("venue"),
        year=meta.get("year"),
        identifiers=Identifiers(
            doi=ids.get("doi"),
            arxiv_id=ids.get("arxiv_id"),
            url=ids.get("url"),
        ),
    )


def process_dataset(
    input_path: str,
    output_path: str,
    download_dir: str = "./down",
    resume: bool = True,
    save_every: int = 100,
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data["instances"]
    total = len(instances)
    start_idx = 0

    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_instances = existing.get("instances", [])
        processed_count = len(
            [inst for inst in existing_instances if inst.get("retrieval_result")]
        )
        if processed_count > 0:
            print(
                f"Resuming: {processed_count}/{total} instances already have retrieval_result"
            )
            data = existing
            instances = data["instances"]
            start_idx = processed_count

    os.makedirs(download_dir, exist_ok=True)

    print(f"Processing {total} instances, starting at index {start_idx}")
    print(f"Download directory: {os.path.abspath(download_dir)}")
    print(f"Output: {output_path}")

    try:
        from tqdm import tqdm

        iterator = tqdm(range(start_idx, total), initial=start_idx, total=total)
    except ImportError:

        def iterator_wrapper():
            for i in range(start_idx, total):
                print(f"[{i+1}/{total}]", end=" ", flush=True)
                yield i

        iterator = iterator_wrapper()

    start_time = time.time()
    for i in iterator:
        instance = instances[i]
        if instance.get("retrieval_result"):
            continue

        try:
            citation = instance_to_citation(instance, i)
            result = retrieve_paper(citation, directory=download_dir)
            instance["retrieval_result"] = result
        except Exception as e:
            instance["retrieval_result"] = {
                "ref_id": uuid.uuid4().hex[:12],
                "exists": False,
                "source": None,
                "title": instance.get("citation_metadata", {}).get("title"),
                "year": instance.get("citation_metadata", {}).get("year"),
                "authors": instance.get("citation_metadata", {}).get("authors", []),
                "identifiers": instance.get("citation_metadata", {}).get(
                    "identifiers", {}
                ),
                "file_path": None,
                "error": str(e),
            }

        if (i + 1) % save_every == 0 or (i + 1) == total:
            data["metadata"]["num_instances"] = len(instances)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({total / elapsed:.1f} instances/s)")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve references for each instance in a citation dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON dataset path (e.g., citation_dataset_xxx_out.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: input path with _retrieved suffix)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./down",
        help="Directory to store downloaded PDFs (default: ./down)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output file (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume behavior",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N instances (default: 100)",
    )
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_retrieved{ext}"

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        download_dir=args.download_dir,
        resume=args.resume,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()


# python reference_retrieval.py --input citation_dataset_20260621_111023_out.json --output citation_dataset_20260621_111023_retrieved.json --download-dir ./down --resume --save-every 200