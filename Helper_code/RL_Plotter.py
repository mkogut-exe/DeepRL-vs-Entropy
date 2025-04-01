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
    df['Reward_MA'] = df['Reward'].rolling(window=window, min_periods=1).mean()

    # Create figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Plot actor loss
    ax1.plot(df['Episode'], df['Actor_Loss'], 'b-', alpha=0.3)
    ax1.plot(df['Episode'], df['Actor_Loss_MA'], 'b-', linewidth=2, label='Actor Loss')
    ax1.set_ylabel('Actor Loss')
    ax1.set_title('Wordle RL Training Metrics')
    ax1.legend()
    ax1.grid(True)

    # Set y-axis limits for actor loss (ignore outliers)
    q80_actor = df['Actor_Loss'].quantile(0.80)
    ax1.set_ylim(-4,0.3)#q80_actor * 1.6)

    # Plot critic loss
    ax2.plot(df['Episode'], df['Critic_Loss'], 'r-', alpha=0.3)
    ax2.plot(df['Episode'], df['Critic_Loss_MA'], 'r-', linewidth=2, label='Critic Loss')
    ax2.set_ylabel('Critic Loss')
    ax2.legend()
    ax2.grid(True)

    # Set y-axis limits for critic loss (ignore outliers)
    q80_critic = df['Critic_Loss'].quantile(0.80)
    ax2.set_ylim(0, q80_critic * 1.4)

    # Plot win rate
    ax3.plot(df['Episode'], df['Win_Rate'], 'g-', alpha=0.3)
    ax3.plot(df['Episode'], df['Win_Rate_MA'], 'g-', linewidth=2, label='Win Rate')
    ax3.set_ylabel('Win Rate')
    ax3.grid(True)

    # Add baseline for random guessing
    ax3.axhline(y=0.81, color='gray', linestyle='--', alpha=0.7, label='Random Baseline')
    ax3.legend()

    # Set y-axis limits for win rate
    win_max = max(0.3, df['Win_Rate'].max() * 1.05)
    ax3.set_ylim(0.7, win_max)
    
    # Plot rewards
    ax4.plot(df['Episode'], df['Reward'], 'purple', alpha=0.3)
    ax4.plot(df['Episode'], df['Reward_MA'], 'purple', linewidth=2, label='Reward')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    ax4.legend()
    
    # Set y-axis limits for rewards
    reward_min = df['Reward'].min() * 0.99
    reward_max = df['Reward'].max() * 1.01
    ax4.set_ylim(reward_min, reward_max)

    # Save plot with same name as input file but PNG extension
    output_path = f'{os.path.splitext(file_path)[0]}.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as '{output_path}' (based on input file '{file_path}')")
    plt.show()


if __name__ == "__main__":
    plot_training_metrics(file_path='training_metrics_20250331115431_ARLGv1-win_epo-100000_AR-10_CR-2_AS-2x256-Lr-1e-06-Bs-1000.csv')
