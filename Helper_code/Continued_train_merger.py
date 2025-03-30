import pandas as pd
import os


def align_training_episodes(first_file, second_file, output_file=None):
    """
    Aligns the episode numbers of the second training run to continue from where the first run ended.

    Args:
        first_file: Path to the first training run CSV file
        second_file: Path to the second training run CSV file
        output_file: Path to save the merged result (default uses hardcoded name)
    """
    # Add file extension if missing
    if not first_file.endswith('.csv'):
        first_file += '.csv'
    if not second_file.endswith('.csv'):
        second_file += '.csv'

    # Read the CSV files
    df1 = pd.read_csv(first_file)
    df2 = pd.read_csv(second_file)

    # Print column names for debugging
    print(f"First file columns: {df1.columns.tolist()}")
    print(f"Second file columns: {df2.columns.tolist()}")

    # Get the name of the episode column (first column)
    episode_col = df1.columns[0]

    # Ensure second file has the same column structure
    if episode_col not in df2.columns:
        raise ValueError(f"Column '{episode_col}' not found in second file")

    # Find the maximum episode in first run
    max_episode = df1[episode_col].max()

    # Find the minimum episode in second run
    min_episode_second = df2[episode_col].min()

    # Calculate increment between episodes in second run
    increment = df2[episode_col].iloc[1] - df2[episode_col].iloc[0] if len(df2) > 1 else 500

    # Adjust episode numbers in second run
    df2[episode_col] = df2[episode_col] - min_episode_second + max_episode + increment

    # Create merged dataset
    result = pd.concat([df1, df2], ignore_index=True)

    # Save to CSV
    if output_file is None:
        output_file = 'aligned_training_metrics.csv'
    result.to_csv(output_file, index=False)

    print(f"Aligned and merged training data saved to {output_file}")
    print(f"First run ended at episode {max_episode}")
    print(f"Second run starts at episode {max_episode + increment}")

    return result


# Call the function with file paths
align_training_episodes(
    'training_metrics_Rv2_epo-40000_AR-10_CR-2_AS-8x256-Lr-1e-05-Bs-1024.csv',
    'Stats/training_metrics_Rv2_epo-80000_AR-10_CR-2_AS-8x256-Lr-1e-05-Bs-1024.csv'
)