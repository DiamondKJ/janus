# /JANUS-CORE/visualize_dynamics.py

import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_cognitive_dynamics(log_path, interaction_index=-1):
    """
    Plots the blend weight history for a specific interaction from the log.
    
    Args:
        log_path (str): Path to the cognitive_dynamics.jsonl file.
        interaction_index (int): The index of the interaction to plot. 
                                 -1 for the most recent, 0 for the first, etc.
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at '{log_path}'")
        return

    with open(log_path, 'r') as f:
        interactions = [json.loads(line) for line in f]

    if not interactions:
        print("Error: Log file is empty.")
        return

    if interaction_index >= len(interactions) or interaction_index < -len(interactions):
        print(f"Error: Invalid interaction index. Log contains {len(interactions)} interactions.")
        return

    interaction = interactions[interaction_index]
    
    user_prompt = interaction.get('user_prompt', 'N/A')
    blend_history = interaction.get('blend_history', [])
    initial_blend = interaction.get('initial_blend', 0.5)

    if not blend_history:
        print(f"Error: No blend history found for interaction {interaction_index}.")
        return

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(blend_history, marker='o', linestyle='-', markersize=4, label='Live Blend Weight')
    
    # Add horizontal lines for context
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Perfect Balance')
    ax.axhline(y=initial_blend, color='red', linestyle=':', linewidth=2, label=f'Initial Blend ({initial_blend:.2f})')

    # Add labels and title
    ax.set_xlabel("Token Generation Step", fontsize=12)
    ax.set_ylabel("Blend Weight", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['Purely Analytical', 'Leans Analytical', 'Balanced', 'Leans Associative', 'Purely Associative'])
    
    title = ax.set_title(f"Cognitive Flow for Prompt: \"{user_prompt[:50]}...\"", fontsize=14, pad=20)
    plt.setp(title, color='#333333')
    
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    plot_filename = f"cognitive_ecg_interaction_{interaction_index if interaction_index != -1 else len(interactions)-1}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Janus's cognitive dynamics.")
    parser.add_argument(
        "--index", 
        type=int, 
        default=-1, 
        help="The interaction index to plot from the log file (e.g., 0 for first, -1 for last). Default is -1."
    )
    args = parser.parse_args()

    log_file_path = os.path.join(os.path.dirname(__file__), "reports/cognitive_dynamics.jsonl")
    plot_cognitive_dynamics(log_file_path, args.index)