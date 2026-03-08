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

def postprocess_missing_field():
    with open(r'data\UCT_dataset\UCT_all.json', 'r', encoding='utf-8') as f:
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

        with open(r'data\UCT_dataset\fab_instances.json', 'w', encoding='utf-8') as f:
            json.dump(fab_instances, f, ensure_ascii=False, indent=4)

        data["instances"] = filtered_instances
        with open(r'data\UCT_dataset\filter_missing_values.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Total missing field instances: {len(fab_instances)}")

def postprocess_latex():
    with open(r'data\UCT_dataset\filter_missing_values.json', 'r', encoding='utf-8') as f:
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

        data["instances"] = instances
        with open(r'data\UCT_dataset\UCT_all_postprocessed.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # print(instances[3385]["citation_metadata"]["title"])
        # print(converter.latex_to_text(instances[3385]["surrounding_context"]))

if __name__ == "__main__":
    # test_process_id()
    # sweep_multi_field()

    postprocess_latex()
    # postprocess_missing_field()
