import json
from parse_ref_by_arxivID import process_id

if __name__ == "__main__":
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