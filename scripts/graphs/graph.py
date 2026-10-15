import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for eps
eps_data = {
    'epsilon': ['6.6', '16.6', '30'],
    'P Value KGW+': [1.27E-3, 8.83E-8, 3.10E-14],      
    'P Value KTH+': [0.50, 0.49, 0.51],                                      
}

delta_data = {
    'delta': ['3.0', '4.0', '5.0', '6.0'],
    'P Value': [1.27E-3, 1.59E-5, 4.97E-6, 1.45E-6],                                       
}


def create_pvalue_plot(data):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times']  # Fallback options

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.tick_params(axis='y', labelsize=25) 
    ax.tick_params(axis='x', labelsize=25)

    # Set up log scale
    ax.set_yscale('log')
    ax.set_xlabel('Watermark Ratio (%)', fontsize=25, weight='bold')
    ax.set_ylabel('log₁₀(p value)', fontsize=25, weight='bold')
    
    # Plot both p-value series
    ax.plot(df['epsilon'], df['P Value KGW+'], 
            marker='o', markersize=8, color='blue', label='KGW+', linewidth=3)
    ax.plot(df['epsilon'], df['P Value KTH+'], 
            marker='s', markersize=8, color='green', label='KTH+', linewidth=3)
    
    # Set y-axis limits to show both very small and ~0.1 values
    ax.set_ylim(1e-15, 0)
    
    # Create custom y-ticks showing logarithmic values
    y_ticks = [1e-15, 1e-10, 1e-5, 0.50]
    y_tick_labels = [
        '-15', '-10', '-5', '-0.3'
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels) 
    
    # Add horizontal grid lines only at major ticks
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=25, framealpha=1, loc='lower left', prop={'weight': 'bold', 'size': 25})
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig(f'epsilon.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_delta_plot(data):
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.tick_params(axis='y', labelsize=25) 
    ax.tick_params(axis='x', labelsize=25) 

    # Set up log scale
    ax.set_yscale('log')
    ax.set_xlabel('Logits Bias', fontsize=25, weight='bold')
    ax.set_ylabel('log₁₀(p value)', fontsize=25, weight='bold')

    # Plot p-value series for delta
    ax.plot(df['delta'], df['P Value'], 
            marker='o', markersize=8, color='blue', label='KGW+', linewidth=3)
    
    # Set y-axis limits appropriate for the delta data
    ax.set_ylim(10**-6, 10**-2.5)
    
    # Create custom y-ticks showing logarithmic values
    y_ticks = [1e-6, 1e-5, 1e-4, 1e-3]
    y_tick_labels = ['-6', '-5', '-4', '-3']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    # Add horizontal grid lines only at major ticks
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=25, framealpha=1, loc='lower left', prop={'weight': 'bold', 'size': 25})
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig('delta.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots for both datasets
create_pvalue_plot(eps_data)
create_delta_plot(delta_data)