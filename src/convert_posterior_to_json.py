"""
Convert Bayesian posteriors to JSON for web visualization
"""

import pandas as pd
import json
from pathlib import Path

# Configuration
OUTPUT_DIR = Path('data')
POSTERIORS_FILE = OUTPUT_DIR / 'bayesian_posteriors_2023_24.parquet'
JSON_OUTPUT = Path('shot_chart_data.json')

def convert_to_json():
    """Convert parquet data to JSON format for web viz"""
    
    print(f"\n{'='*70}")
    print("CONVERTING DATA FOR WEB VISUALIZATION")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"📂 Loading {POSTERIORS_FILE}...")
    df = pd.read_parquet(POSTERIORS_FILE)
    print(f"✅ Loaded {len(df):,} records for {df['player_id'].nunique()} players\n")
    
    # Convert to list of dictionaries
    data = df.to_dict('records')
    
    # Convert numpy types to native Python types for JSON serialization
    for record in data:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, 'item'):  # numpy scalar
                record[key] = value.item()
    
    # Save to JSON
    print(f"💾 Saving to {JSON_OUTPUT}...")
    with open(JSON_OUTPUT, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size = JSON_OUTPUT.stat().st_size / 1024
    
    print(f"✅ Successfully created {JSON_OUTPUT}")
    print(f"   File size: {file_size:.1f} KB")
    print(f"   Records: {len(data):,}")
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Open shot_chart_visualization.html in a browser")
    print("2. The visualization will load shot_chart_data.json")
    print("3. Search for a player and explore their Bayesian shot chart!")
    print()

if __name__ == "__main__":
    convert_to_json()