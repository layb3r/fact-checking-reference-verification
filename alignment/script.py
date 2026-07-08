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

if __name__ == "__main__":
    json_path = r"..\data_generation\citation_dataset_20260621_111023_out_pdf.json"
    pdfs_path = r"./data/mock"

    # add_pdf_field(json_path)
    filter_instances_having_pdf(json_path, pdfs_path)