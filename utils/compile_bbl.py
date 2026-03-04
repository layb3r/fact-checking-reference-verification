import subprocess
import os
import re

def compile_bibitem(bibitem_string):
    """
    Wraps a \bibitem in a minimal LaTeX doc, compiles it, 
    and extracts the plain text result.
    """
    temp_tex = "temp_item.tex"
    temp_pdf = "temp_item.pdf"
    
    # Minimal LaTeX wrapper
    latex_template = rf"""
    \documentclass{{article}}
    \usepackage[utf8]{{inputenc}}
    \begin{{document}}
    \begin{{thebibliography}}{{1}}
    {bibitem_string}
    \end{{thebibliography}}
    \end{{document}}
    """

    print(latex_template)

    with open(temp_tex, "w", encoding="utf-8") as f:
        f.write(latex_template)

    try:
        # 1. Compile to PDF
        subprocess.run(["pdflatex", "-interaction=nonstopmode", temp_tex], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Extract text from PDF (requires poppler/pdftotext)
        result = subprocess.check_output(["pdftotext", "-layout", temp_pdf, "-"], 
                                         encoding="utf-8")
        
        # Cleanup temp files
        for ext in [".tex", ".pdf", ".aux", ".log"]:
            if os.path.exists("temp_item" + ext):
                os.remove("temp_item" + ext)
        
        # Clean up the extracted text (remove labels like [1])
        cleaned_text = re.sub(r'\[\d+\]\s*', '', result).strip()
        return cleaned_text

    except Exception as e:
        return f"Error compiling item: {e}"

if __name__ == "__main__":
    # Example Usage:
    bibitems = [
        r"\bibitem{zhou2024_layerwise_lmc} Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, and Wei Hu. \newblock Going beyond linear mode connectivity: The layerwise linear feature connectivity. \newblock \emph{Advances in Neural Information Processing Systems}, 36, 2024."
    ]

    mapping = {item: compile_bibitem(item) for item in bibitems}

    for raw, compiled in mapping.items():
        print(f"RAW: {raw[:50]}...")
        print(f"COMPILED: {compiled}\n")
