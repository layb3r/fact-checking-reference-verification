import json
import pandas as pd
import json
from pylatexenc.latex2text import LatexNodes2Text


def sweep_multi_field(in_dir, out_dir, head_num=70):
    df = pd.read_csv(in_dir)
    filtered_df = df.groupby('field').head(120).reset_index(drop=True)
    filtered_df.to_csv(out_dir, index=False)

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


def combined_postprocess(in_dir, failed_dir=None, out_dir=None,
                         require_fields=True, remove_arxiv_venue=True,
                         max_authors=10, validate_claim_context=True):
    """
    Run all postprocessing steps in sequence in a single pass:

      1. Filter instances with missing required fields
      2. Convert LaTeX encoding to plain text
      3. Remove instances with arXiv venue or too many authors
      4. Keep only instances where claim_text appears in surrounding_context

    Args:
        in_dir:        Input JSON file path
        failed_dir:    Output path for removed instances (missing fields)
        out_dir:       Output path for cleaned JSON
        require_fields:       Whether to filter missing fields (Step 1)
        remove_arxiv_venue:   Whether to filter arXiv venue entries (Step 3)
        max_authors:          Maximum allowed authors (Step 3, default 10)
        validate_claim_context: Whether to validate claim in context (Step 4)
    """

    failed_dir = failed_dir if failed_dir is not None else in_dir[:-5] + '_failed.json'
    out_dir = out_dir if out_dir is not None else in_dir[:-5] + '_out.json'

    with open(in_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    instances = data["instances"]
    converter = LatexNodes2Text()

    # Step 1: filter missing fields
    if require_fields:
        filtered = [
            inst for inst in instances
            if inst["citation_metadata"]["title"]
            and inst["citation_metadata"]["venue"]
            and inst["citation_metadata"]["year"]
            and len(inst["citation_metadata"]["authors"]) > 0
        ]
        removed = [
            inst for inst in instances
            if not inst["citation_metadata"]["title"]
            or not inst["citation_metadata"]["venue"]
            or not inst["citation_metadata"]["year"]
            or len(inst["citation_metadata"]["authors"]) == 0
        ]
        with open(failed_dir, 'w', encoding='utf-8') as f:
            json.dump(removed, f, ensure_ascii=False, indent=4)
        print(f"Step 1: removed {len(removed)} instances with missing fields")
        instances = filtered

    # Step 2: convert LaTeX to plain text; drop instances where conversion fails
    converted = []
    dropped = 0
    for inst in instances:
        try:
            inst["claim_text"] = converter.latex_to_text(inst["claim_text"])
            inst["surrounding_context"] = converter.latex_to_text(inst["surrounding_context"])
            inst["citation_metadata"]["title"] = converter.latex_to_text(inst["citation_metadata"]["title"])
            inst["citation_metadata"]["authors"] = [
                converter.latex_to_text(a) for a in inst["citation_metadata"]["authors"]
            ]
            inst["citation_metadata"]["venue"] = converter.latex_to_text(inst["citation_metadata"]["venue"])
            converted.append(inst)
        except Exception:
            dropped += 1
    instances = converted
    print(f"Step 2: converted LaTeX for {len(instances)} instances ({dropped} dropped due to conversion error)")

    # Step 3: remove arXiv venue and long author lists
    if remove_arxiv_venue:
        before = len(instances)
        instances = [
            inst for inst in instances
            if "arxiv" not in inst["citation_metadata"]["venue"].lower()
            and len(inst["citation_metadata"]["authors"]) < max_authors
        ]
        print(f"Step 3: removed {before - len(instances)} instances (arXiv venue or >= {max_authors} authors)")

    # Step 4: validate claim_text is within surrounding_context
    if validate_claim_context:
        before = len(instances)
        instances = [inst for inst in instances if inst["claim_text"] in inst["surrounding_context"]]
        print(f"Step 4: removed {before - len(instances)} instances where claim_text not in surrounding_context")

    data["instances"] = instances
    data["metadata"]["num_instances"] = len(instances)
    with open(out_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Done: {len(instances)} instances saved to {out_dir}")


if __name__ == "__main__":
    # sweep_multi_field(r'.\arxiv_papers_20260305_214804.csv', r'.\arxiv-collection-120.csv')
    combined_postprocess(r".\citation_dataset_20260624_105140.json")