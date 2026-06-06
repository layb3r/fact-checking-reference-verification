import json
# from parse_ref_by_arxivID import process_id
# import pandas as pd
import json
from pylatexenc.latex2text import LatexNodes2Text

def clean_latex(data):
    converter = LatexNodes2Text()
    if isinstance(data, dict):
        return {k: clean_latex(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_latex(i) for i in data]
    elif isinstance(data, str):
        # Normalizes escaped backslashes (\\\"a -> \"a) before converting
        return converter.latex_to_text(data)
    return data

def test_process_id():
    arxiv_id = '2603.03973v1'
    output = 'test_output'
    result = process_id(arxiv_id, output)

    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    # with open('test.json', 'r', encoding='utf-8') as f:
    #     test = json.load(f)
    #     for instance in test["instances"]:
    #         if(instance["claim_text"] == ""):
    #             print(instance)

def sweep_multi_field():
    df = pd.read_csv(r'D:\Research\Fact-Checking\fact-checking-reference-verification\data\multi-field-papers\arxiv-only-collection\arxiv_papers_20260305_214804.csv')
    filtered_df = df.groupby('field').head(40).reset_index(drop=True)
    filtered_df.to_csv(r"D:\Research\Fact-Checking\fact-checking-reference-verification\data\multi-field-papers\arxiv-only-collection\arxiv-collection-40.csv", index=False)

def postprocess_missing_field(in_dir, failed_dir, out_dir):
    with open(in_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]
        filtered_instances = [
            inst for inst in instances 
            if inst["citation_metadata"]["title"] 
            and inst["citation_metadata"]["venue"] 
            and inst["citation_metadata"]["year"] 
            and len(inst["citation_metadata"]["authors"]) > 0
        ]

        fab_instances = [
            inst for inst in instances 
            if not inst["citation_metadata"]["title"] 
            or not inst["citation_metadata"]["venue"] 
            or not inst["citation_metadata"]["year"] 
            or len(inst["citation_metadata"]["authors"]) == 0
        ] 

        with open(failed_dir, 'w', encoding='utf-8') as f:
            json.dump(fab_instances, f, ensure_ascii=False, indent=4)

        data["instances"] = filtered_instances
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Total missing field instances: {len(fab_instances)}")

def postprocess_latex():
    with open(r'data\UCT_dataset\UCT_all_postprocessed_2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]
        # print(len(instances))
        # print(instances[0]["surrounding_context"])
        converter = LatexNodes2Text()

        for i, instance in enumerate(instances):
            print(f"Processed {i} instances ---")
            instance["claim_text"] = converter.latex_to_text(instance["claim_text"])
            instance["surrounding_context"] = converter.latex_to_text(instance["surrounding_context"])
            instance["citation_metadata"]["title"] = converter.latex_to_text(instance["citation_metadata"]["title"])
            instance["citation_metadata"]["authors"] = [
                converter.latex_to_text(author) for author in instance["citation_metadata"]["authors"]
            ]
            instance["citation_metadata"]["venue"] = converter.latex_to_text(instance["citation_metadata"]["venue"])

        data["instances"] = instances
        with open(r'data\UCT_dataset\UCT_all_postprocessed_latex.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # print(instances[3385]["citation_metadata"]["title"])
        # print(converter.latex_to_text(instances[3385]["surrounding_context"]))

def hotfix_venue():
    with open(r'data\UCT_dataset\UCT_all_postprocessed.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]
        converter = LatexNodes2Text()
        for instance in instances:
            instance["citation_metadata"]["venue"] = converter.latex_to_text(instance["citation_metadata"]["venue"])

        data["instances"] = instances
        with open(r'data\UCT_dataset\UCT_all_postprocessed.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def clean_some_arxiv_and_too_long_authors():
    with open(r'data\UCT_dataset\UCT_all_postprocessed_new.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]
        # print(len([inst for inst in instances if "arxiv" in inst["citation_metadata"]["venue"].lower() ]))

        instances = [
            inst for inst in instances if "arxiv" not in inst["citation_metadata"]["venue"].lower() and len(inst["citation_metadata"]["authors"]) < 10
        ]

        data["instances"] = instances

        # print(len(instances))
        with open(r'data\UCT_dataset\UCT_all_postprocessed.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def check_claim_text(in_dir, out_dir):
    with open(in_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]
        fixed = [inst for inst in instances if inst["claim_text"] in inst["surrounding_context"]]

        data["instances"] = fixed
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def filter_by_arxiv_id(in_dir, out_dir):
    import re
    import requests

    ARXIV_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

    with open(in_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instances = data["instances"]
    kept = []

    for i, inst in enumerate(instances):
        identifiers = inst.get("citation_metadata", {}).get("identifiers", {})
        arxiv_id = identifiers.get("arxiv_id")

        if not arxiv_id or not ARXIV_PATTERN.match(str(arxiv_id)):
            continue

        url = identifiers.get("url")
        if not url:
            url = f"https://arxiv.org/abs/{arxiv_id}"
            identifiers["url"] = url

        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200:
                continue
        except requests.RequestException:
            continue

        kept.append(inst)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(instances)}")

    data["instances"] = kept
    with open(out_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Kept {len(kept)}/{len(instances)} instances with valid arxiv IDs")


def filter_arxiv_instances(current_data_dir, out_dir=None, save_every=50):
    import requests
    import time
    import xml.etree.ElementTree as ET
    import os

    ARXIV_API = "https://export.arxiv.org/api/query"
    ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

    with open(current_data_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
        instances = data["instances"]

    out_dir = out_dir or current_data_dir.replace(".json", "_arxiv_filtered.json")
    checkpoint_file = out_dir.replace(".json", "_checkpoint.json")
    fail_file = out_dir.replace(".json", "_fails.json")

    total = len(instances)

    # --- resume from checkpoint ---
    start_idx = 0
    filtered = []
    fails = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)
            start_idx = ckpt["last_index"] + 1
        if os.path.exists(out_dir):
            with open(out_dir, 'r', encoding='utf-8') as f:
                filtered = json.load(f).get("instances", [])
        if os.path.exists(fail_file):
            with open(fail_file, 'r', encoding='utf-8') as f:
                fails = json.load(f)
        print(f"Resumed from instance {start_idx}/{total} ({len(filtered)} kept, {len(fails)} failed)")

    # --- helper: periodic save ---
    def save_progress(idx):
        ckpt_data = {"last_index": idx}
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(ckpt_data, f)

        out = dict(data)
        out["instances"] = filtered
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        with open(fail_file, 'w', encoding='utf-8') as f:
            json.dump(fails, f, ensure_ascii=False, indent=4)

        print(f"  Checkpoint saved at instance {idx + 1}/{total}")

    # --- main loop ---
    for i in range(start_idx, total):
        inst = instances[i]
        meta = inst.get("citation_metadata", {})
        arxiv_id = meta.get("identifiers", {}).get("arxiv_id")
        title = meta.get("title")

        found = False

        # 1) Try arxiv_id lookup
        if arxiv_id:
            clean_id = arxiv_id.replace("arXiv:", "").strip()
            parts = clean_id.split("v")
            query_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else clean_id

            try:
                resp = requests.get(
                    ARXIV_API,
                    params={"id_list": query_id, "max_results": 1},
                    headers={"User-Agent": "fact-checking-reference-verification/1.0"},
                    timeout=30,
                )
                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ARXIV_NS)
                if entries and any(
                    link.attrib.get("title") == "pdf"
                    for link in entries[0].findall("atom:link", ARXIV_NS)
                ):
                    filtered.append(inst)
                    found = True
            except Exception:
                pass

        # 2) Fallback: search by title
        if not found and title:
            try:
                resp = requests.get(
                    ARXIV_API,
                    params={"search_query": f'ti:"{title}"', "max_results": 1},
                    headers={"User-Agent": "fact-checking-reference-verification/1.0"},
                    timeout=30,
                )
                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", ARXIV_NS)
                if entries and any(
                    link.attrib.get("title") == "pdf"
                    for link in entries[0].findall("atom:link", ARXIV_NS)
                ):
                    id_el = entries[0].find("atom:id", ARXIV_NS)
                    if id_el is not None and id_el.text and "/abs/" in id_el.text:
                        inst["citation_metadata"]["identifiers"]["arxiv_id"] = id_el.text.split("/abs/")[-1]
                    filtered.append(inst)
                    found = True
            except Exception:
                pass

        if not found:
            fails.append(inst)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total}")

        if (i + 1) % save_every == 0 or i == total - 1:
            save_progress(i)

        time.sleep(1)

    print(f"Done. arxiv+pdf instances: {len(filtered)}, fails: {len(fails)}")
    print(f"Saved to: {out_dir}")


import random

def sample_and_merge(path_1, path_2, num_1, num_2, out_dir=None):
    with open(path_1, 'r', encoding='utf-8') as f:
        data_1 = json.load(f)
    with open(path_2, 'r', encoding='utf-8') as f:
        data_2 = json.load(f)

    sampled_1 = random.sample(data_1["instances"], min(num_1, len(data_1["instances"])))
    sampled_2 = random.sample(data_2["instances"], min(num_2, len(data_2["instances"])))

    merged = {"instances": sampled_1 + sampled_2}

    if out_dir:
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(merged['instances'])} instances to {out_dir}")

    return merged


if __name__ == "__main__":
    # in_dir = r'data\UCT_dataset\UCT_all_postprocessed_latex.json'
    # failed_dir = r'bin\filtered-conf\_fails_postprocessed_after_cleaning_venue.json'
    # out_dir = r'data\UCT_dataset\UCT_all_postprocessed_new.json'
    # test_process_id()
    # sweep_multi_field()

    # postprocess_latex()
    # postprocess_missing_field(in_dir, failed_dir, out_dir)
    # hotfix_venue()

    # clean_some_arxiv_and_too_long_authors()
    # check_claim_text(
    #     r"data\UCT_dataset\UCT_all_postprocessed_new_filtered.json",
    #     r"data\UCT_dataset\UCT_all_postprocessed_new_filtered_2.json"
    # )

    current_data_dir = r"..\data\UCT_dataset\UCT_all_postprocessed_new_filtered_2.json"
    # filter_arxiv_instances(current_data_dir)
    # filter_by_arxiv_id(current_data_dir, r"..\data\UCT_dataset\UCT_arxiv.json")

    sample_and_merge(
        r"..\data\UCT_dataset\UCT_arxiv.json",
        r".\negative-instances-generator\negative_alignments.json",
        10,
        19,
        r"merged.json"
    )
# {
#   "claim_text": "... [CITATION] ...",
#   "surrounding_context": "...",
#   "citation_metadata": {
#     "title": "...",
#     "authors": ["..."],
#     "venue": "...",
#     "year": 2023,
#     "identifiers": {
#       "doi": null,
#       "arxiv_id": null,
#       "url": null
#     }
#   },
#   "true_outputs": {
#         "true_alignment": 0,
#     ...
#   }
# }