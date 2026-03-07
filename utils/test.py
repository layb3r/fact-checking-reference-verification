import json
from parse_ref_by_arxivID import process_id
import pandas as pd

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

    pass
if __name__ == "__main__":
    # test_process_id()
    sweep_multi_field()
