"""
CLI entry point: chạy bằng
  python -m extract_references_type <đường_dẫn_file_pdf> [grobid_url]
"""

import json
import sys
from pathlib import Path

from grobid import GROBIDReferenceExtractor


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách sử dụng: python -m extract_references_type <đường_dẫn_file_pdf> [grobid_url]")
        print("\nVí dụ:")
        print("  python -m extract_references_type paper.pdf")
        print("  python -m extract_references_type paper.pdf http://localhost:8070")
        sys.exit(1)

    pdf_path = sys.argv[1]
    grobid_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8070"

    print(f"Đang xử lý file: {pdf_path}")
    print(f"GROBID server: {grobid_url}")

    extractor = GROBIDReferenceExtractor(grobid_url)

    try:
        # Bước 1: Lấy raw text từ PDF
        print("\n[Bước 1] Trích xuất raw references từ PDF...")
        references = extractor.extract_references_from_pdf(pdf_path)

        # Tách các references bị gộp
        references = extractor.split_merged_references(references)

        extractor.print_references(references)

        # Lưu raw references
        raw_output = Path(pdf_path).stem + "_references_raw.json"
        with open(raw_output, "w", encoding="utf-8") as f:
            json.dump(references, f, ensure_ascii=False, indent=2)
        print(f"Raw references đã lưu vào: {raw_output}")

        # Bước 2: Parse từng raw citation thành structured data
        print(f"\n[Bước 2] Parse {len(references)} citations qua /api/processCitation...")
        flat_references = extractor.parse_all_citations(references)

        # Bước 3: Chuyển đổi sang nested format
        print("\n[Bước 3] Chuyển đổi sang nested format...")
        parsed_references = [
            extractor.format_output_schema(flat, idx)
            for idx, flat in enumerate(flat_references, 1)
        ]

        # Lưu parsed references
        parsed_output = "tests/json/" + Path(pdf_path).stem + "_references_parsed_type.json"
        with open(parsed_output, "w", encoding="utf-8") as f:
            json.dump(parsed_references, f, ensure_ascii=False, indent=2)
        print(f"\nParsed references đã lưu vào: {parsed_output}")

    except Exception as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()