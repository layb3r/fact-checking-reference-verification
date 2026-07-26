import json
import os

def add_pdf_field(json_path : str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for idx, instance in enumerate(data["instances"]):
            instance["existence_retrieval"] = {
                "pdf_path": str(idx) + ".pdf"
            }

        out_dir = json_path[:-5] + "_pdf.json"
        print(out_dir)
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def filter_instances_having_pdf(json_path: str, pdfs_path: str):
    # list files in pdfs_path
    pdf_files = set(os.listdir(pdfs_path))
    print(f"Found {len(pdf_files)} PDF files in {pdfs_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        filtered_instances = []
        for instance in data["instances"]:
            pdf_path = instance.get("existence_retrieval", {}).get("pdf_path")
            if pdf_path and os.path.exists(os.path.join(pdfs_path, pdf_path)):
                filtered_instances.append(instance)

        data["instances"] = filtered_instances

        out_dir = json_path[:-5] + "_filtered.json"
        print(out_dir)
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def test():
    path = r".\data\citation_dataset_270_mapped.json"
    path2 = r"..\data_generation\citation_dataset_270.json"
    out_dir = r".\data\citation_dataset_270_add_pdf.json"
    with open(path, 'r', encoding='utf-8') as f:
        with open(path2, 'r', encoding='utf-8') as f1:
            data2 = json.load(f1)
            data = json.load(f)

            for instance1, instance2 in zip(data, data2["instances"]):
                instance2["citation_metadata"]["filepaths"] = instance1["citation_metadata"]["filepaths"]

            with open(out_dir, 'w', encoding='utf-8') as f2:
                json.dump(data2, f2, ensure_ascii=False, indent=4)
            # print(len(data))
            # not_exist = 0

            # for instance in data:
            #     # if instance doesn't have filepaths entry
            #     if "filepaths" not in instance["citation_metadata"] or len(instance["citation_metadata"]["filepaths"]) == 0:
            #         not_exist = not_exist + 1
            #         # print(instance["citation_metadata"]["title"])

            # print(not_exist)

def filter_pdf():
    path = r".\data\citation_dataset_270_add_pdf.json"
    out_dir = r".\data\citation_dataset_270_add_pdf_filtered.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # print(len(data))
        not_exist = 0

        data["instances"] = [inst for inst in data["instances"] if ("filepaths" in inst["citation_metadata"] and len(inst["citation_metadata"]["filepaths"]) != 0)]
        print(len(data["instances"]))

        with open(out_dir, 'w', encoding='utf-8') as f2:
            json.dump(data["instances"], f2, ensure_ascii=False, indent=4)

# write a function to for loop through \data\citation_dataset_270_add_pdf_filtered.json to check the filepath exists and openable:
# note that root data path is .\data\final, moreover the filepaths are like '.\pdf_arxiv\...pdf'
# if we os.join like usual, might result in '.\data\final\./pdf_doi/10.1109_WACV51458.2022.00264.pdf'
def check_pdf_openable():
    path = r".\data\citation_dataset_270_add_pdf_filtered.json"
    out_dir = r".\data\citation_dataset_270_add_pdf_filtered_successful.json"

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        not_exist = 0
        not_openable = 0
        successful_instances = 0

        for instance in data:
            filepaths = instance["citation_metadata"]["filepaths"]

            # try until there is 1 file that is openable, if none of the files are openable, then count it as not openable

            openable = False
            for filepath in filepaths:
                clean_filepath = os.path.normpath(filepath).replace("/", "\\") 
                full_path = os.path.join(r".\data\final", clean_filepath)
                # print(full_path)
                if not os.path.exists(full_path):
                    not_exist += 1
                    continue

                try:
                    with open(full_path, 'rb') as f:
                        f.read(1)
                    openable = True
                    successful_instances += 1
                    break
                except Exception as e:
                    not_openable += 1
                    continue

        # we filter to another dataset containing only the successful instances
        # and also the filepaths contain only 1 openable file, if there are multiple files, we only keep the first openable one
        filtered_instances = []
        for instance in data:
            filepaths = instance["citation_metadata"]["filepaths"]
            openable_filepaths = []
            for filepath in filepaths:
                clean_filepath = os.path.normpath(filepath).replace("/", "\\") 
                full_path = os.path.join(r".\data\final", clean_filepath)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            f.read(1)
                        openable_filepaths.append(filepath)
                        break  # Only keep the first openable file
                    except Exception as e:
                        continue
            if len(openable_filepaths) > 0:
                instance["citation_metadata"]["filepaths"] = [openable_filepaths[0]]  # Keep only the first openable file
                filtered_instances.append(instance)
            

        print(f"Total files that do not exist: {not_exist}")
        print(f"Total files that are not openable: {not_openable}")
        print(f"Total successful instances: {successful_instances}/{len(data)}")

        # Save the filtered instances to a new JSON file
        with open(out_dir, 'w', encoding='utf-8') as f:
            json.dump(filtered_instances, f, ensure_ascii=False, indent=4)

def count_number_of_instances(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Total number of instances: {len(data['instances'] if 'instances' in data else data)}")

# check if all instances has field retrieved_evidences and in that has field extractive_chunks which is a non-empty list:
def check_retrieved_evidences(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:

        num_failed_instances = 0
        data = json.load(f)
        for instance in data["instances"]:
            if "retrieved_evidences" not in instance or "extractive_chunks" not in instance["retrieved_evidences"] or len(instance["retrieved_evidences"]["extractive_chunks"]) == 0:
                num_failed_instances += 1
                print(f"Instance {instance} does not have retrieved_evidences or extractive_chunks is empty")
                
    print(f"Total number of instances that failed the check: {num_failed_instances}/{len(data['instances'])}")

REMOVE_FIELDS = {"is_adversarial", "adversarial_metadata", "instance_id"}


def _interleave_multi(groups):
    """Deterministically interleave multiple groups so each is spread evenly."""
    import itertools

    groups = [list(g) for g in groups]
    total = sum(len(g) for g in groups)
    if total == 0:
        return []

    # Remove empty groups
    groups = [g for g in groups if g]

    weights = [1.0 / len(g) for g in groups]
    accs = weights[:]
    indices = [0] * len(groups)
    result = []

    while True:
        active = [(i, accs[i]) for i in range(len(groups)) if indices[i] < len(groups[i])]
        if not active:
            break
        # Pick the group with the smallest accumulator
        i = min(active, key=lambda x: x[1])[0]
        result.append(groups[i][indices[i]])
        indices[i] += 1
        if indices[i] < len(groups[i]):
            accs[i] += weights[i]

    return result


def combine_stratified(
    enriched_path: str,
    negatives_path: str,
    output_path: str,
):
    with open(enriched_path, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)
    enriched_instances = enriched_data.get("instances", enriched_data if isinstance(enriched_data, list) else [])

    with open(negatives_path, "r", encoding="utf-8") as f:
        neg_data = json.load(f)
    neg_instances = neg_data if isinstance(neg_data, list) else neg_data.get("instances", neg_data if isinstance(neg_data, list) else [])

    all_instances = enriched_instances + neg_instances
    for inst in all_instances:
        for field in REMOVE_FIELDS:
            inst.pop(field, None)

    # Group by true_alignment
    groups = {}
    for inst in all_instances:
        label = inst.get("true_outputs", {}).get("true_alignment")
        groups.setdefault(label, []).append(inst)

    print("Label distribution:")
    for label, items in sorted(groups.items()):
        print(f"  label {label}: {len(items)}")

    combined = _interleave_multi(list(groups.values()))

    print(f"Total: {len(combined)} instances")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {output_path}")


LABEL_TO_NUM = {
    "SUPPORTED": 0,
    "UNSUPPORTED": 2,
    "UNCERTAIN": 3,
    "UNSURE": 3,
    "PARTIALLY": 1
}


def fix_negatives_labels(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # instances = data.get("adversarial_instances", data.get("instances", []))
    instances = data if isinstance(data, list) else data.get("instances", [])
    for inst in instances:
        meta = inst.get("adversarial_metadata", {})
        target = meta.get("target_alignment_label")
        if target is not None:
            inst.setdefault("true_outputs", {})["true_alignment"] = LABEL_TO_NUM[target]

    out_path = json_path.replace(".json", "_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed {len(instances)} instances -> {out_path}")

DRIFT_TO_NUM = {
    "over_claim": 1,
    "context_shift": 2,
    "reversal": 2,
    "tangential": 3,
}

LABEL_TO_NUM2 = {
    "SUPPORTED": 0,
    "UNSUPPORTED": 2,
    "UNCERTAIN": 3,
}


def fix_negatives_labels_2(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # instances = data.get("adversarial_instances", data.get("instances", []))
    instances = data if isinstance(data, list) else data.get("instances", [])
    for inst in instances:
        meta = inst.get("adversarial_metadata", {})
        drift = meta.get("drift_type")
        target = meta.get("target_alignment_label")
        if drift is not None and drift == 'over_claim':
            inst.setdefault("true_outputs", {})["true_alignment"] = DRIFT_TO_NUM[drift]
        if drift is not None and drift != 'over_claim':
            inst.setdefault("true_outputs", {})["true_alignment"] = LABEL_TO_NUM2[target]

    out_path = json_path.replace(".json", "_fixed_2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed {len(instances)} instances via drift_type -> {out_path}")

def combine_datasets(path_1: str, path_2: str, output_path: str):
    import random
    with open(path_1, "r", encoding="utf-8") as f:
        data_1 = json.load(f)
    data1_instances = data_1.get("instances", data_1 if isinstance(data_1, list) else [])

    with open(path_2, "r", encoding="utf-8") as f:
        data_2 = json.load(f)
    data2_instances = data_2.get("instances", data_2 if isinstance(data_2, list) else [])

    all_instances = data1_instances + data2_instances

    random.shuffle(all_instances)
    print(f"Total: {len(all_instances)} instances")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {output_path}")

# remove instances where [CITATION] is not in claim_text field:
def remove_instances_without_citation(json_path: str, output_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data if isinstance(data, list) else data.get("instances", data if isinstance(data, list) else [])
    filtered_instances = [inst for inst in instances if "[CITATION]" in inst.get("claim_text", "")]

    print(f"Removed {len(instances) - len(filtered_instances)} instances without [CITATION] in claim_text")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_instances, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {output_path}")

def combine_negatives(
    partially_path: str,
    uncertain_path: str,
    negatives_path: str,
    output_path: str,
):
    """Combine neg_partially, neg_uncertain, and filtered reversal/context_shift UNSUPPORTED negatives."""

    def load_instances(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            return d
        for key in ("adversarial_instances", "instances"):
            if key in d:
                return d[key]
        return []

    partially = load_instances(partially_path)
    uncertain = load_instances(uncertain_path)

    negatives = load_instances(negatives_path)
    filtered_neg = []
    for inst in negatives:
        meta = inst.get("adversarial_metadata", {})
        label = meta.get("target_alignment_label")
        drift = meta.get("drift_type")
        if label == "UNSUPPORTED" and drift in ("reversal", "context_shift"):
            filtered_neg.append(inst)

    # for inst in partially + uncertain + filtered_neg:
    #     for field in REMOVE_FIELDS:
    #         inst.pop(field, None)

    groups = [partially, uncertain, filtered_neg]
    combined = _interleave_multi(groups)

    print(f"Partially: {len(partially)} + Uncertain: {len(uncertain)} + UNSUPPORTED reversal/context_shift: {len(filtered_neg)}")
    print(f"Total: {len(combined)} -> {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    json_path = r"..\data_generation\citation_dataset_20260621_111023_out_pdf.json"
    pdfs_path = r"./data/mock"

    # add_pdf_field(json_path)
    # filter_instances_having_pdf(json_path, pdfs_path)
    # test()
    # filter_pdf()
    # check_pdf_openable()
    # count_number_of_instances(r".\data\citation_dataset_270_add_pdf_filtered_successful.json")
    # count_number_of_instances(r".\data\results.json")
    # check_retrieved_evidences(r".\data\results.json")
    # fix_negatives_labels(r".\data\neg_test.json")
    # fix_negatives_labels_2(r".\data\negatives_added_over_claim_with_citation_corrected.json")
    combine_stratified(
        r".\data\enriched.json",
        r".\data\neg_test_fixed.json",
        r".\data\final_1_test.json"
    )

    # count_number_of_instances(r".\data\combined_added.json")
    # combine_datasets(
    #     r".\data\negatives.json",
    #     r".\data\negatives_over_claim.json",
    #     r".\data\negatives_added_over_claim.json"
    # )

    # combine_negatives(
    #     r".\data\neg_partially.json",
    #     r".\data\neg_uncertain.json",
    #     r".\data\negatives.json",
    #     r".\data\neg_filtered_combined.json",
    # )


    # count_number_of_instances(r".\data\negatives.json")
    # count_number_of_instances(r".\data\negatives_over_claim.json")
    # count_number_of_instances(r".\data\negatives_added_over_claim.json")

    # remove_instances_without_citation(
    #     r".\data\neg_filtered_combined.json",
    #     r".\data\neg_test.json"
    # )