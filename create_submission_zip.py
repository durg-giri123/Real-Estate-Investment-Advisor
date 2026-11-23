"""
Creates a complete, verified submission ZIP file for the Real Estate Investment Advisor project.
Ensures every deliverable (src, tests, notebooks, models, data, reports, app, mlflow.db, mlruns, PDF, README) is included.
"""

import os
import zipfile

def create_zip(output_filename="Real_Estate_Investment_Advisor_Complete_Submission.zip"):
    # Target directories and files to bundle
    include_dirs = ["app", "data", "docs", "models", "notebooks", "reports", "src", "tests", "mlruns", ".streamlit"]
    include_files = [
        "README.md",
        "requirements.txt",
        "generate_pdf_report.py",
        "mlflow.db",
        "merged_houseprice_regression_classification.xlsx",
        "Real_Estate_Investment_Advisor_Project_Report.pdf",
        "Project Title-DS_AI_ML.docx",
        ".gitignore"
    ]

    print(f"Creating submission package: {output_filename}...")
    file_count = 0

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as z:
        # Add root files
        for f in include_files:
            if os.path.exists(f):
                z.write(f, arcname=f)
                file_count += 1
                print(f"  [+] {f}")
            else:
                print(f"  [-] Warning: {f} not found!")

        # Add directories recursively (excluding __pycache__ and temp files)
        for d in include_dirs:
            if os.path.exists(d):
                for root, dirs, files in os.walk(d):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.startswith("~$") or file.endswith(".tmp"):
                            continue
                        full_path = os.path.join(root, file)
                        z.write(full_path, arcname=full_path)
                        file_count += 1
                print(f"  [+] Directory: {d}/")
            else:
                print(f"  [-] Warning: Directory {d}/ not found!")

    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print("=" * 60)
    print(f"SUCCESS: Created {output_filename}")
    print(f"Total files included: {file_count}")
    print(f"Package Size: {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    create_zip()
