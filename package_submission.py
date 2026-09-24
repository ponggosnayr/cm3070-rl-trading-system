"""
Automated Clean Packaging Script for CM3070 Final Project Submission.
Packages only necessary source code, models, data, tests, documentation, and pre-built frontend.
Strictly excludes virtual environments, node_modules, cache dirs, temporary logs, and credentials.
"""

import os
import zipfile
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
OUTPUT_ZIP = ROOT_DIR / "CM3070_Final_Project_Submission.zip"

EXCLUDE_DIRS = {
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".gemini",
    ".agents",
    "tmp",
    "scratch",
    "archive",
    "tensorboard_logs"
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".bak",
    ".aux",
    ".log",
    ".out",
    ".toc",
    ".lof",
    ".lot",
    ".mp4"  # Videos are submitted separately or via URL link
}

EXCLUDE_FILES = {
    ".env",
    "CM3070_Final_Project_Submission.zip",
    "draftdraft.pdf",
    "draftreport.pdf",
    "texput.log",
    # Exclude backup and redundant model weights (keep active persistent_brain.pth and normalizer)
    "persistent_brain.pth.bak",
    "persistent_brain_final.pth",
    "persistent_brain_daily_backup.pth",
    "persistent_brain_old5act_backup.pth",
    "persistent_brain_normalizer.pkl.bak",
    # Exclude draft / historical PDFs in docs to keep archive clean (FINAL_PROJECT_REPORT.pdf is preserved)
    "DRAFT_FINAL_PROJECT_REPORT.pdf",
    "DRAFT_FINAL_PROJECT_REPORT_V2.pdf",
    "PRELIMINARY_PROJECT_REPORT.pdf"
}

def should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Check directory exclusions
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith("."):
            return True
            
    # Check file exclusions
    filename = rel_path.name
    if filename in EXCLUDE_FILES:
        return True
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True
        
    return False

def package_submission():
    print(f"Creating submission package: {OUTPUT_ZIP.name}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(ROOT_DIR)
                
                if should_exclude(rel_path):
                    continue
                    
                zipf.write(full_path, arcname=str(rel_path))
                file_count += 1
                total_uncompressed_bytes += full_path.stat().st_size

    zip_size = OUTPUT_ZIP.stat().st_size
    print(f"Package created successfully!")
    print(f"Total files archived: {file_count}")
    print(f"Uncompressed size: {total_uncompressed_bytes / (1024*1024):.2f} MB")
    print(f"Compressed ZIP size: {zip_size / (1024*1024):.2f} MB")
    print(f"Location: {OUTPUT_ZIP}")

if __name__ == "__main__":
    package_submission()
