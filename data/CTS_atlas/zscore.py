import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# DATA_FOLDER = path to the data
DATA_FOLDER = '/path/data'

# Load the data
df = pd.read_csv(os.path.join(DATA_FOLDER, 'deconvo_results.txt'), sep='\t')

def zscore_normalization(df: pd.DataFrame, out_filepath: str) -> None:
    
    # Identify controls and cases
    mask = df.index.str.contains('Control')
    controls = df[mask]
    cases = df[~mask]

    # Compute minimum standard deviation
    stds = np.asarray([controls[entity].std() for entity in df.columns])
    min_nonzero_std = np.min(stds[stds > 0]) * 0.5
    
    # Check each case for values outside the normality range
    results = []
    
    for sample in cases.index:
        sample_result = []
        for entity in df.columns:
            
            value = cases.loc[sample, entity]
            mean_control = controls[entity].mean()
            std_control = max(controls[entity].std(), min_nonzero_std)
            
            # Calculate the z-score
            z_score = (value - mean_control) / std_control

            if abs(z_score) >= 2:
                deviation = f'{abs(z_score):.2f}SD {"higher" if (z_score > 0) else "lower"}'
            else:
                continue 
    
            sample_result.append({
                'Entity': entity,
                'Value': value,
                'Z_Score': z_score,
                'Deviation': deviation
            })
    
        # Add the sample results to the overall results --> open in results=[]
        results.extend({
            'Sample': sample,
            **res
        } for res in sample_result)
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a csv file
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    results_df.to_csv(out_filepath, sep='\t', index=False)
    
    print(f"Z-scores saved to '{out_filepath}'")
