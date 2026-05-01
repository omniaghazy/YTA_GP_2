import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(output_path="seagate_cleaned_cols.csv", num_drives=200, days=120):
    start_date = datetime(2025, 9, 1)
    data = []
    
    smart_cols = [
        "smart_1_raw", "smart_5_raw", "smart_7_raw",
        "smart_187_raw", "smart_197_raw", "smart_9_normalized",
        "smart_190_normalized", "smart_193_normalized"
    ]
    
    for i in range(num_drives):
        sn = f"SN_{i:03d}"
        model = "ST_MODEL"
        failure_day = np.random.randint(10, days) if i % 4 == 0 else None 
        # Spread failures across the whole range to ensure Val/Test sets have samples
        
        for d in range(days):
            current_date = start_date + timedelta(days=d)
            is_failure = 1 if failure_day == d else 0
            
            row = {
                "date": current_date.strftime("%Y-%m-%d"),
                "serial_number": sn,
                "model": model,
                "failure": is_failure,
            }
            
            # Add some dummy SMART values
            for col in smart_cols:
                row[col] = np.random.normal(100, 10)
            
            data.append(row)
            if is_failure:
                break # Drive failed, no more rows
                
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated dummy data with {len(df)} rows at {output_path}")

if __name__ == "__main__":
    generate_dummy_data()
