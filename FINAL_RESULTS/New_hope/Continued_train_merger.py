import pandas as pd
import os
import datetime


def align_training_episodes(first_file, second_file, output_file=None, episode_col=None):
    """
    Aligns the episode numbers of the second training run to continue from where the first run ended.

    Args:
        first_file: Path to the first training run CSV file
        second_file: Path to the second training run CSV file
        output_file: Path to save the merged result (default uses hardcoded name)
        episode_col: Name of the episode column (if None, uses first column)
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

    # Get the name of the episode column (first column if not specified)
    if episode_col is None:
        episode_col = df1.columns[0]
        print(f"Using {episode_col} as the episode column")

    # Ensure both files have the episode column
    if episode_col not in df1.columns:
        raise ValueError(f"Column '{episode_col}' not found in first file")
    if episode_col not in df2.columns:
        raise ValueError(f"Column '{episode_col}' not found in second file")
    
    # Check if column structures match
    if len(df1.columns) != len(df2.columns):
        print(f"Warning: Files have different number of columns ({len(df1.columns)} vs {len(df2.columns)})")
    
    # Verify columns match (except possibly the episode column)
    non_episode_cols1 = [col for col in df1.columns if col != episode_col]
    non_episode_cols2 = [col for col in df2.columns if col != episode_col]
    if non_episode_cols1 != non_episode_cols2:
        print("Warning: Column names differ between files (excluding episode column)")
        print(f"First file: {non_episode_cols1}")
        print(f"Second file: {non_episode_cols2}")

    # Find the maximum episode in first run
    max_episode = df1[episode_col].max()

    # Find the minimum episode in second run
    min_episode_second = df2[episode_col].min()

    # Calculate increment between episodes in second run
    if len(df2) > 1:
        increment = df2[episode_col].iloc[1] - df2[episode_col].iloc[0]
    else:
        # Default increment if second file has only one row
        increment = 500
        print(f"Warning: Second file has only one row. Using default increment of {increment}")

    # Adjust episode numbers in second run
    df2[episode_col] = df2[episode_col] - min_episode_second + max_episode + increment

    # Create merged dataset
    result = pd.concat([df1, df2], ignore_index=True)

    # Save to CSV
    if output_file is None:
        # Use a short filename with a timestamp to avoid Windows path length issues
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'aligned_{timestamp}.csv'

    # Ensure output has .csv extension
    if not output_file.endswith('.csv'):
        output_file += '.csv'

    # Always save in the current directory to avoid FileNotFoundError due to missing folders
    output_file = os.path.basename(output_file)

    result.to_csv(output_file, index=False)

    print(f"Aligned and merged training data saved to {output_file}")
    print(f"First run ended at episode {max_episode}")
    print(f"Second run starts at episode {max_episode + increment}")

    return result


if __name__ == "__main__":
    # Call the function with file paths - update these paths as needed
    align_training_episodes(
        'aligned_20250715_112909',
        'training_metrics_FINAL_20250714120755_WV-WR0-win_epo-200000_AR-10_CR-2_AS-7x256-Lr-1e-07-Bs-5000'
    )
    
    # Alternatively, to specify a different episode column:
    # align_training_episodes(
    #     'first_file',
    #     'second_file',
    #     episode_col='Episode'
    # )
