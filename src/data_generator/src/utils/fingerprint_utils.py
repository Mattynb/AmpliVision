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
