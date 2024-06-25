from matplotlib import markers
import matplotlib.pyplot as plt
from scipy.stats import norm

colors = ['r', 'g', 'b']

def limit_std(fingerprint: dict, limit: int = 15) -> dict:
    for color in colors:
        for i in range(1, 4, 2):
            if fingerprint[color][i] > limit:
                fingerprint[color][i] = limit
    return fingerprint

def format_line(line: list, spot: int) -> list:
    spot = (spot * 3) 
    return [float(line[spot + 13].strip()), float(line[spot + 14].strip()), float(line[spot + 15].strip())]

def display_fingerprint(fingerprint:map, title_sufx:str='')->None:
    """"""

    for block_type, values in fingerprint.items():
        title = f'{block_type} fingerprint {title_sufx}'

        fig, ax = plt.subplots(3, 1, figsize=(10, 10), num=title)

        for i, color in enumerate(colors):
            mean1, std1, mean2, std2 = values[color]
            x = range(0, 256)
            ax[i].plot(x, norm.pdf(x, mean1, std1), color=color, label='spot1')
            ax[i].plot(x, norm.pdf(x, mean2, std2), color=color, linestyle='dashed', label='spot2')
            ax[i].set_title(f'{color} distribution')
            ax[i].legend()
        plt.show()



def visualize_fingerprints_with_colors(fingerprints):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('RGB Values for Different Blocks with Actual Colors', fontsize=16)
    
    block_names = list(fingerprints.keys())
    
    for idx, (block, ax) in enumerate(zip(block_names, axes.flatten())):
        data = fingerprints[block]
        x_labels = ['Spot 1', 'Spot 2']
        
        for i in range(2):
            r_mean = 255-data['r'][i * 2]
            g_mean = 255-data['g'][i * 2]
            b_mean = 255-data['b'][i * 2]
            r_stdev = data['r'][i * 2 + 1]
            g_stdev = data['g'][i * 2 + 1]
            b_stdev = data['b'][i * 2 + 1]
            
            color = (r_mean / 255, g_mean / 255, b_mean / 255)
            ax.errorbar(x_labels[i], r_mean, yerr=r_stdev, fmt='o', color='r', capsize=20)
            ax.errorbar(x_labels[i], g_mean, yerr=g_stdev, fmt='o', color='g', capsize=20)
            ax.errorbar(x_labels[i], b_mean, yerr=b_stdev, fmt='o', color='b', capsize=20)
            
            ax.scatter([x_labels[i]], [r_mean], color=color, s=100, marker='D', edgecolor='r', zorder=3)
            ax.scatter([x_labels[i]], [g_mean], color=color, s=100, marker='D', edgecolor='g', zorder=3)
            ax.scatter([x_labels[i]], [b_mean], color=color, s=100, marker='D', edgecolor='b', zorder=3)
        
        ax.set_title(block)
        ax.set_ylim(0, 255)  # Adjust the y-limit as needed
        ax.set_ylabel('Value')
        ax.legend(['R Mean ± Std', 'G Mean ± Std', 'B Mean ± Std'])
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

"""
Fingerprints:
'Control Block': 
{
    'r': [2.625, 2.9553976043842223, 108.375, 3.276335605520289], 
    'g': [2.375, 2.6896793489187516, 138.5, 3.570714214271425], 
    'b': [2.875, 2.666341125962693, 133.875, 3.515590277606308]
}, 
'Test Block 3': 
{   
    'r': [52.625, 3.119995993587171, 95.5, 2.5], 
    'g': [95.625, 2.4462982238476156, 129.5, 2.449489742783178], 
    'b': [94.5, 2.345207879911715, 126.75, 2.436698586202241]
}, 
'Test Block 2': 
{
    'r': [69.5, 1.8708286933869707, 96.25, 3.7332961307670196], 
    'g': [74.0, 2.0615528128088303, 101.5, 2.9154759474226504], 
    'b': [20.875, 1.4523687548277813, 58.5, 2.598076211353316]
}, 
'Test Block 1': 
{
    'r': [38.25, 2.6339134382131846, 92.875, 4.075460096725276], 
    'g': [80.75, 2.436698586202241, 122.5, 4.330127018922194], 
    'b': [80.375, 3.0388114452858046, 117.625, 5.146297212559725]
}
"""