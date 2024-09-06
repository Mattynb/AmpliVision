"Unweighted graph class meant to represent the test structure. E.g Sample pad -> test block 1 -> etc.."

import networkx as nx
import matplotlib.pyplot as plt


class TestGraph:
    def __init__(self, blocks):
        self.graph = self.build_graph(blocks)

    def build_graph(self, blocks):
        G = nx.DiGraph()

        # add nodes (block types) to graph.
        nodes = [block.block_type for block in blocks]

        # make sure duplicate names are changed. Ex [wick, wick] -> [wick_1, wick_2]
        nodes = self.make_unique_names(nodes)
  
        G.add_nodes_from(nodes)
        # add edges (sequence of blocks) to graph
        positions = [block.index for block in blocks]
        
        try:
            head = nodes.index('sample_block')
        except ValueError:
            # block identification error
            return G
        
        edges = self.get_edges(nodes, positions, head)
        G.add_edges_from(edges)

        return G

    def get_edges(self, nodes: list, positions: list, head_node_index: int) -> list:
        """Finds the sequence of blocks. eg. sample -> test 1 -> test2 -> wick block"""

        # Initialize the search
        edges = []
        head_node = nodes.pop(head_node_index)
        head_pos = positions.pop(head_node_index)

        # Start the recursive search for edges
        edges.extend(self.search_edges(head_node, head_pos, nodes, positions))

        return edges

    def search_edges(self, current_node, current_pos, remaining_nodes, remaining_positions):
        edges = []
        y, x = current_pos

        # Define potential neighbors
        neighbors = [
            (y, x + 1),
            (y, x - 1),
            (y + 1, x),
            (y - 1, x)
        ]

        for neighbor in neighbors:

            # list the grid positions. Ex: (5, 2)

            if neighbor in remaining_positions:

                # append found edge
                neighbor_index = remaining_positions.index(neighbor)
                remaining_positions.remove(neighbor)

                neighbor_node = remaining_nodes.pop(neighbor_index)
                edges.append((current_node, neighbor_node))

                # recursively search for neighboring edges and add to list
                edges.extend(
                    self.search_edges(
                        neighbor_node, neighbor, remaining_nodes, remaining_positions
                    )
                )

        return edges

    def display(self):
        G = self.graph
        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
                font_size=15, font_weight="bold", arrows=True)
        plt.show()

    def make_unique_names(self, nodes):
        count = {}
        unique_names = []
        for name in nodes:
            if name in count:
                count[name] += 1
                unique_name = f"{name}_{count[name]}"
            else:
                count[name] = 0
                unique_name = name
            unique_names.append(unique_name)
        return unique_names

    def example_usage(self):

        # add nodes
        sequence = [1, 2, 3, 4]
        G = nx.DiGraph()
        G.add_nodes_from(sequence)

        # add edges
        edges = [(1, 2), (2, 3), (1, 4)]
        G.add_edges_from(edges)

        # Draw the graph
        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
                font_size=15, font_weight="bold", arrows=True)

        # how to get graph information
        print(G.nodes)
        print(G.out_edges)

        plt.show()


if __name__ == '__main__':
    TG = TestGraph()
    TG.example_usage()
