# check_project.py - Place in your project root
import sys
import os
from pathlib import Path

def check_all():
    print("=" * 60)
    print("PROJECT STATUS CHECK")
    print("=" * 60)
    
    checks = []
    
    # 1. Check Python version
    python_version = sys.version_info
    checks.append(("Python Version", f"{python_version.major}.{python_version.minor}.{python_version.micro}"))
    
    # 2. Check virtual environment
    is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    checks.append(("Virtual Environment", "✅ Active" if is_venv else "❌ Not active"))
    
    # 3. Check current directory
    current_dir = Path.cwd()
    checks.append(("Project Directory", str(current_dir)))
    
    # 4. Check project structure
    required_folders = ["data/raw", "src"]
    for folder in required_folders:
        folder_path = Path(folder)
        checks.append((f"Folder: {folder}", "✅ Exists" if folder_path.exists() else "❌ Missing"))
    
    # 5. Check key files
    required_files = [
        "data/raw/Fake.csv",
        "data/raw/True.csv", 
        "src/data_loader.py",
        "src/preprocessing.py",
        "src/train.py"
    ]
    
    for file in required_files:
        file_path = Path(file)
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            checks.append((f"File: {file}", f"✅ Exists ({size_kb:.1f} KB)"))
        else:
            checks.append((f"File: {file}", "❌ Missing"))
    
    # 6. Check datasets
    data_dir = Path("data/raw")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        checks.append(("CSV Datasets", f"✅ {len(csv_files)} files found"))
        for csv_file in csv_files[:3]:  # Show first 3
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            checks.append((f"  - {csv_file.name}", f"{size_mb:.1f} MB"))
    else:
        checks.append(("CSV Datasets", "❌ No data folder"))
    
    # 7. Check dependencies
    dependencies = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn", "sklearn"),
        ("joblib", "joblib"),
        ("nltk", "nltk"),
    ]
    
    for dep_name, dep_alias in dependencies:
        try:
            if dep_alias == "pd":
                import pandas as pd
                version = pd.__version__
            elif dep_alias == "np":
                import numpy as np
                version = np.__version__
            elif dep_alias == "sklearn":
                import sklearn
                version = sklearn.__version__
            elif dep_alias == "joblib":
                import joblib
                version = joblib.__version__
            elif dep_alias == "nltk":
                import nltk
                version = nltk.__version__
            checks.append((f"Dependency: {dep_name}", f"✅ {version}"))
        except ImportError:
            checks.append((f"Dependency: {dep_name}", "❌ Not installed"))
    
    # Display results
    print("\n" + "=" * 60)
    print("CHECK RESULTS")
    print("=" * 60)
    
    all_good = True
    max_label_length = max(len(label) for label, _ in checks)
    
    for label, status in checks:
        padding = " " * (max_label_length - len(label))
        print(f"{label}:{padding} {status}")
        if "❌" in status:
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ PROJECT IS READY TO RUN!")
        print("\nNext steps:")
        print("1. Run: python train_working.py  (to train model)")
        print("2. Run: python predict_simple.py (to test predictions)")
        print("3. Run: python demo.py           (for FYP presentation)")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print("\nPlease fix the issues marked with ❌ above.")
    
    return all_good

if __name__ == "__main__":
    check_all()