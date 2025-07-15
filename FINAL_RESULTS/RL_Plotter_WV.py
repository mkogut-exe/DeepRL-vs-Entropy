import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_metrics(file_path='training_metrics.csv'):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)

    # Calculate moving averages for smoother plots
    window = 3
    df['Actor_Loss_MA'] = df['Actor_Loss'].rolling(window=window, min_periods=1).mean()
    df['Critic_Loss_MA'] = df['Critic_Loss'].rolling(window=window, min_periods=1).mean()
    df['Win_Rate_MA'] = df['Win_Rate'].rolling(window=window, min_periods=1).mean()

    # Create figure with four subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # Plot actor loss
    ax1.plot(df['Episode'], df['Actor_Loss'], 'b-', alpha=0.3)
    ax1.plot(df['Episode'], df['Actor_Loss_MA'], 'b-', linewidth=2, label='Actor Loss')
    ax1.set_ylabel('Actor Loss')
    ax1.set_title('Wordle WV-WR0 Training Metrics')
    ax1.legend()
    ax1.grid(True)

    # Set y-axis limits for actor loss to always include min and max values with 10% margin
    actor_min = df['Actor_Loss'].min() - abs(df['Actor_Loss'].min()) * 0.10 -0.015
    actor_max = df['Actor_Loss'].max() + abs(df['Actor_Loss'].max()) * 0.10 +0.015
    ax1.set_ylim(actor_min, actor_max)

    # Plot critic loss
    ax2.plot(df['Episode'], df['Critic_Loss'], 'r-', alpha=0.3)
    ax2.plot(df['Episode'], df['Critic_Loss_MA'], 'r-', linewidth=2, label='Critic Loss')
    ax2.set_ylabel('Critic Loss')
    ax2.legend()
    ax2.grid(True)

    # Set y-axis limits for actor loss to always include min and max values with 10% margin
    actor_min = df['Critic_Loss'].min() - abs(df['Actor_Loss'].min()) * 0.10 -0.015
    actor_max = df['Critic_Loss'].max() + abs(df['Actor_Loss'].max()) * 0.10 +0.015
    ax2.set_ylim(actor_min, actor_max)

    # Plot win rate
    ax3.plot(df['Episode'], df['Win_Rate'], 'g-', alpha=0.3)
    ax3.plot(df['Episode'], df['Win_Rate_MA'], 'g-', linewidth=2, label='Win Rate')
    ax3.set_ylabel('Win Rate')
    ax3.grid(True)


    # Add baseline for random guessing
    # random 0.4229
    # entropy maximizer 0.8022
    ax3.axhline(y=0.4229, color='gray', linestyle='--', alpha=0.7, label='Random Baseline Win Rate')
    ax3.legend()
    ax3.axhline(y=0.8022, color='#1ff0ff',lw=2, linestyle='-', alpha=1, label='Entropy Maximizer Win Rate')
    ax3.legend()

    # Adjust y-axis limits to ensure data is visible
    win_min = max(0, df['Win_Rate'].min() * 0.95)   # Lower bound with 5% margin
    win_max = min(1, df['Win_Rate'].max() * 1.05)  # Upper bound with 5% margin
    ax3.set_ylim(win_min, win_max)


    # Save plot with same name as input file but PNG extension
    output_path = f'{os.path.splitext(file_path)[0]}.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as '{output_path}' (based on input file '{file_path}')")
    plt.show()


if __name__ == "__main__":
    plot_training_metrics(file_path='aligned_20250712_130430_FINAL_WV-WR0-win_epo-2000000_AR-10_CR-2_AS-7x256-Lr-1e-05-Bs-5000.csv')
