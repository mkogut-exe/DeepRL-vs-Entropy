import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_win_rates_from_multiple_files(csv_files, output_path = 'out.png',EM = False):
    plt.figure(figsize=(12, 8))

    all_dfs = []
    for file_info in csv_files:
        file_path = file_info["path"]
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            continue
        df = pd.read_csv(file_path)
        all_dfs.append((df, file_info["label"]))

    min_win_rate = 1.0
    max_win_rate = 0.0

    for df, label in all_dfs:
        min_win_rate = min(min_win_rate, df['Win_Rate'].min())
        max_win_rate = max(max_win_rate, df['Win_Rate'].max())

        window = 3
        df['Win_Rate_MA'] = df['Win_Rate'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['Episode'], df['Win_Rate_MA'], linewidth=2, label=label)

    plt.ylabel('Win Rate')
    plt.xlabel('Episode')

    plt.grid(True)

    # Add baselines
    plt.axhline(y=0.4229, color='gray', linestyle='--', alpha=0.7, label='Random Baseline Win Rate')
    if EM:
        plt.axhline(y=0.8022, color='#1ff0ff',lw=2, linestyle='-', alpha=1, label='Entropy Maximizer Win Rate')

    plt.legend()

    win_min = max(0, min_win_rate * 0.95)
    win_max = min(1, max_win_rate * 1.05)
    plt.ylim(win_min, win_max)


    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as '{output_path}'")
    plt.show()

if __name__ == "__main__":
    # List of CSV files to plot. Add more entries to this list to plot more results.
    output_path_all = 'all_win_rates_400k.png'
    plot_all = [
        {
            "path": "training_metrics_FINAL_20250704141427_WV-WR0-win_epo-400000_AR-10_CR-2_AS-7x256-Lr-1e-05-Bs-5000.csv",
            "label": "WV-WR0"
        },
        {
            "path": "training_metrics_FINAL_20250705182425_WV-WR+-1-win_epo-400000_AR-10_CR-2_AS-7x256-Lr-1e-05-Bs-5000.csv",
            "label": "WV-WR0"
        },
        {
            "path": "training_metrics_FINAL_20250518120513_non-ARLG-IR-win_epo-200000_AR-10_CR-2_AS-4x256-Lr-1e-05-Bs-5000.csv",
            "label": "non-ARLG-IR"
        },
        {
            "path": "training_metrics_FINAL_20250708155926_non-ARLG-WR0-win_epo-200000_AR-10_CR-2_AS-4x256-Lr-1e-05-Bs-5000.csv",
            "label": "non-ARLG-WR0"
        },
        {
            "path": "aligned_training_metrics_FINAL_20250518223616_ARLG-IR-win_epo-win_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000.csv",
            "label": "ARLG-IR"
        },
        {
            "path": "training_metrics_FINAL_20250522184804_ARLG-IR-wd-win_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000-Dec--0.005.csv",
            "label": "ARLG-IR-wd"
        },
        {
            "path": "training_metrics_FINAL_20250522185037_ARLG-WR+-1-win_only_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000.csv",
            "label": "ARLG-WR+-1"
        }

    ]
    output_path_WV400k = 'output_path_WV400k.png'
    plot_WV400k = [
        {
            "path": "training_metrics_FINAL_20250704141427_WV-WR0-win_epo-400000_AR-10_CR-2_AS-7x256-Lr-1e-05-Bs-5000.csv",
            "label": "WV-WR0"
        },
        {
            "path": "training_metrics_FINAL_20250705182425_WV-WR+-1-win_epo-400000_AR-10_CR-2_AS-7x256-Lr-1e-05-Bs-5000.csv",
            "label": "WV-WR+-1"
        }


    ]
    output_path_LG400k = 'output_path_LG400k.png'
    plot_LG400k = [
        {
            "path": "training_metrics_FINAL_20250518120513_non-ARLG-IR-win_epo-200000_AR-10_CR-2_AS-4x256-Lr-1e-05-Bs-5000.csv",
            "label": "non-ARLG-IR"
        },
        {
            "path": "aligned_training_metrics_FINAL_20250518223616_ARLG-IR-win_epo-win_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000.csv",
            "label": "ARLG-IR"
        },
        {
            "path": "training_metrics_FINAL_20250522184804_ARLG-IR-wd-win_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000-Dec--0.005.csv",
            "label": "ARLG-IR-wd"
        },
        {
            "path": "training_metrics_FINAL_20250522185037_ARLG-WR+-1-win_only_epo-400000_AR-10_CR-2_AS-1x256-Lr-1e-05-Bs-5000.csv",
            "label": "ARLG-WR+-1"
        }
    ]
    output_path_WV2400k = 'WV2400k.png'
    plot_WV2400k = [
        {
            "path": "aligned_20250715_11295220250714120755_WV-WR0-win_epo-2400000_AR-10_CR-2_AS-7x256-Lr-2mil_1e-5-200k_1e-Bs-5000.csv",
            "label": "WV-WR0"
        }

    ]

    plot_win_rates_from_multiple_files(plot_WV2400k , output_path_WV2400k, EM=1)
