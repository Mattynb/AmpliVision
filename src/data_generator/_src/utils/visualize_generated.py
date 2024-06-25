from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np

def visualize_generated_pts(data:dict[list[float]], subtract:int = 0)->None:
    fig, ax = plt.subplots()
    num_types = len(data)
    num_pairs = max(len(pairs) for pairs in data.values())

    # Set grid dimensions
    grid_width = 4  # Number of pairs per row
    grid_height = (num_pairs // grid_width) + 1  # Number of rows

   
    for i, (dtype, pairs) in enumerate(data.items()):
        for j, pair in enumerate(pairs):
            row = j // grid_width
            col = j % grid_width

            data_list = [x for x in pair]
            if subtract != 0:
                data_list = [subtract - x for x in data_list]
            r1, g1, b1, r2, g2, b2 = data_list
            
            ax.add_patch(plt.Rectangle((col * 2, i * grid_height + row), 1, 1, color=[r1/255, g1/255, b1/255]))
            ax.add_patch(plt.Rectangle((col * 2 + 1, i * grid_height + row), 1, 1, color=[r2/255, g2/255, b2/255]))

    # Customizing the plot
    ax.set_xlim(-0.1, grid_width * 2)
    ax.set_ylim(-0.1, num_types * grid_height)
    ax.set_yticks(np.arange(num_types * grid_height) + 0.5)
    ax.set_yticklabels([dtype for dtype in data.keys() for _ in range(grid_height)])
    ax.set_xticks([])
    plt.title("RGB Color Pairs Grid")

    plt.show()


def print_generated_pts(pts:dict[list[float]], subtract:int =0)->None:
    print("\n","-"*10, "GENERATED DATA","-"*10,)
    for type in pts:
        print(f"{type}: ")
        for r in pts[type]:
            
            if subtract != 0:
                r = [subtract - x for x in r]

            print(f"\t{r}")
    
    print("-"*30)


def plot_3d_generated_pts(pts:dict[list[float]], title:str = "")->None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    markers = ['o', 's',  'h', 'D']
    colors = ['r', 'g', 'b', 'c']
    seen_types = ()
    legend_handles = []
    for i, type in enumerate(pts):
        for row in pts[type]:
                        
            r1, g1, b1, r2, g2, b2 = row
            ax.scatter(r1, g1, b1, color=[r1/255, g1/255, b1/255], s=100, marker=markers[0], edgecolors=colors[i])
            ax.scatter(r2, g2, b2, color=[r2/255, g2/255, b2/255], s=100, marker=markers[1], edgecolors=colors[i])

            # Check if type has been seen before, so we can assign a different marker
            if type not in seen_types:
                seen_types += (type,)
                legend_handles.append(plt.Line2D([0], [0], marker=markers[0], color=colors[i], markerfacecolor='w', markersize=5, label=f'{type} spot1'))
                legend_handles.append(plt.Line2D([0], [0], marker=markers[1], color=colors[i], markerfacecolor='w', markersize=5, label=f'{type} spot2'))
    

            
    # set max values  for each axis at 255
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)    
    
    # legend 
    ax.legend(handles=legend_handles, loc='best')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()