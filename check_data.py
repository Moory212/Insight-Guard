# check_data.py
import pandas as pd
from pathlib import Path

print("=" * 60)
print("DATA CHECK")
print("=" * 60)

data_dir = Path("data/raw")

if not data_dir.exists():
    print("❌ data/raw directory doesn't exist!")
    exit()

csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    print("❌ No CSV files found in data/raw")
    exit()

print(f"\nFound {len(csv_files)} CSV files:\n")

for i, csv_file in enumerate(csv_files, 1):
    print(f"{i}. {csv_file.name}")
    
    try:
        # Try to read the first few rows
        df = pd.read_csv(csv_file, nrows=3)
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Shape: {df.shape} (rows x columns)")
        print(f"   Columns: {list(df.columns)}")
        
        # Show first row sample
        if len(df) > 0:
            first_row = df.iloc[0]
            print("   First row sample:")
            for col in df.columns[:3]:  # Show first 3 columns
                val = str(first_row[col])
                if len(val) > 50:
                    val = val[:47] + "..."
                print(f"     {col}: {val}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Error reading: {e}")
        print()

# Check for Fake.csv and True.csv specifically
fake_exists = (data_dir / "Fake.csv").exists()
true_exists = (data_dir / "True.csv").exists()

if fake_exists and true_exists:
    print("✅ Found main datasets: Fake.csv and True.csv")
else:
    print("⚠️  Missing main datasets. Training may be limited.")

print("\n" + "=" * 60)