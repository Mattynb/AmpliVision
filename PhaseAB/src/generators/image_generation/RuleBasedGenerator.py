from networkx import DiGraph


class RuleBasedGenerator:
    def __init__(self, graphs: DiGraph, results: dict[dict[dict[list[int]]]], config=None):
        """"""

        self.results = self.validate_results(results)
        self.graphs = self.validate_graphs(graphs)
        self.test_components_path = "PhaseAB/data/test_components"

    def validate_graphs(self, graphs):
        "graphs should all be the same. Warn user if different"
        return graphs

    def validate_results(self, results):
        "results should be all different. Warn user if same"
        return results

    def generate(self, image):
        # Generate image based on rules

        # will need to be broken into functions but the idea is:
        """
        - For each test structure
            - For each target disease
                - For each block in the structure
                    - select the correct image
                    - paint spots according to results
                    - ex: block.paint_spot(mean, std)
                    - place sample block in a randomish position, 
                      place others accordingly. Check for rotation here.
        """
        return image
