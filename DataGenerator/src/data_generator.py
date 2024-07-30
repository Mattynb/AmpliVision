"""Module generates random data points based on the combined fingerprints of the block types."""

import numpy as np


class DataGenerator:
    """Generates random data points based on the combined fingerprints of the block types."""

    def __init__(self, combined_fingerprints: dict):
        self.combined_fingerprints = combined_fingerprints
        self.n_spots = 2
        self.n_block_types = len(combined_fingerprints)

    def generate_points(self, n_points: int) -> dict[list[list[float]]]:
        """
        Generates n random points for each block type.

        Args:
            n_points (int): Number of points to generate.

        Returns:
            dict: Randomly generated points for each block type.

        Example return value:
        {
            "type1": [[r1, g1, b1, r2, g2, b2], ... ],
            "type2": [[r1, g1, b1, r2, g2, b2], ... ],
            ...
        }
        """

        # limit the number of points to the multiple of the top 3 smallest standard deviations
        n_points = self._max_n_points(n_points)

        generated_points = {}
        for blocktype in self.combined_fingerprints:
            pts = self._generate_points_by_blocktype(blocktype, n_points)
            generated_points[blocktype] = pts
        return generated_points

    def _generate_points_by_blocktype(self, block_type: str, n_points: int) -> list[list[list[int]]]:
        """
        Generates n random spot1 & spot2 points based on the block type's spot1 & spot2 normal distribution.

        Args:
            block_type (str): The type of block.
            n_points (int): Number of points to generate.

        Returns:
            list[list[list[int]]]: Randomly generated points for each RGB channel.

        Example return value:
            [[spot1_r [],  spot2_r []], [spot1_g [],  spot2_g []], [spot1_b [],  spot2_b []]]
        """

        triplet_inputs = []
        for rgb in ['r', 'g', 'b']:
            triplet_inputs.append(
                [self.combined_fingerprints[block_type][rgb]])
            # rgb_mean1, rgb_std1, rgb_mean2, rgb_std2 = triplet_inputs[0][0]

        # Generate n random unique triplet points (rgb) for spot1 and spot2
        generated = self._generate_unique_triplet_points(
            n_points, triplet_inputs)

        return generated

    def _generate_unique_triplet_points(self, n_points: int, triplet_inputs: list[list[list[float]]]) -> list[list[list[int]]]:
        """
        Should Return:
        [
            [spot1_r1, spot1_g1, spot1_b1, spot2_r1, spot2_g1, spot2_b1],
            [spot1_r2, spot1_g2, spot1_b2, spot2_r2, spot2_g2, spot2_b2],
            [spot1_r3, spot1_g3, spot1_b3, spot2_r3, spot2_g3, spot2_b3],
            ...
        ]
        """

        # Generate n random unique triplet points (rgb) for spot1 and spot2
        spot1_points = set()
        while len(spot1_points) < n_points:
            point = []
            for t_input in triplet_inputs:
                rgb_mean1, rgb_std1, _, _ = t_input[0]
                point.append(self._generate_point(rgb_mean1, rgb_std1))
            spot1_points.add(tuple(point))
        spot1_points = [list(point) for point in spot1_points]

        spot2_points = set()
        while len(spot2_points) < n_points:
            point = []
            for t_input in triplet_inputs:
                _, _, rgb_mean2, rgb_std2 = t_input[0]
                point.append(self._generate_point(rgb_mean2, rgb_std2))
            spot2_points.add(tuple(point))
        spot2_points = [list(point) for point in spot2_points]

        return [spot1 + spot2 for spot1, spot2 in zip(spot1_points, spot2_points)]

    def _generate_point(self, mean: float, std: float) -> int:
        """
        Generates a random point based on the mean and standard deviation.

        Args:
            mean (float): Mean value.
            std (float): Standard deviation.

        Returns:
            int: Randomly generated point, ensured to be within the RGB range (0-255).
        """
        point = int(np.random.normal(mean, std))
        return max(0, min(255, point))

    def _max_n_points(self, n_points: int) -> int:

        # find the top 3 smallest standard deviations in the block types
        smallest_std_trio = 256*256*256
        for blocktype in self.combined_fingerprints:
            trio1 = np.prod([
                self.combined_fingerprints[blocktype]['r'][1],
                self.combined_fingerprints[blocktype]['g'][1],
                self.combined_fingerprints[blocktype]['b'][1]
            ])
            trio2 = np.prod([
                self.combined_fingerprints[blocktype]['r'][3],
                self.combined_fingerprints[blocktype]['g'][3],
                self.combined_fingerprints[blocktype]['b'][3]
            ])
            trio = min(trio1, trio2)

            smallest_std_trio = min(smallest_std_trio, trio)

        if smallest_std_trio < n_points:
            print(
                f"""Warning: The number of points was limited to {
                    smallest_std_trio} to ensure the uniqueness of points."""
            )
        return min(smallest_std_trio, n_points)

    def _separate_points(self, points: list):
        """
        Transforms points to a flattened structure.

        Args:
            points (list): List of generated points.

        Returns:
            list[list[float]]: Transformed points.

        Example:

        points =
        [
            [spot1_r [...],  spot2_r [...]]
            [spot1_g [...],  spot2_g [...]]
            [spot1_b [...],  spot2_b [...]]
        ]

        separated =
        [
            [spot1_r1, spot1_g1, spot1_b1, spot2_r1, spot2_g1, spot2_b1],
            [spot1_r2, spot1_g2, spot1_b2, spot2_r2, spot2_g2, spot2_b2],
            [spot1_r3, spot1_g3, spot1_b3, spot2_r3, spot2_g3, spot2_b3],
            ...
        ]
        """

        n = len(points[0][0])
        separated = []
        for i in range(n):

            # points[r,g, or b][spot 1 or 2][row i]
            r1 = points[0][0][i]
            g1 = points[1][0][i]
            b1 = points[2][0][i]

            r2 = points[0][1][i]
            g2 = points[1][1][i]
            b2 = points[2][1][i]

            separated.append([r1, g1, b1, r2, g2, b2])

        return separated


if __name__ == '__main__':

    # Test data
    combined = {
        'Test1': {
            'r': [-0.25, 3.031088913245535, 18.5, 10.0],
            'g': [0.75, 2.384848003542364, 90.375, 10.0],
            'b': [-0.125, 3.822967336793113, 56.375, 10.0]
        },
        'Test2': {
            'r': [29.75, 10.0, 69.5, 10.0],
            'g': [61.25, 10.0, 94.75, 10.0],
            'b': [57.125, 10.0, 89.875, 10.0]
        }
    }

    dg = DataGenerator(combined)
    print(dg.generate_n_points(10))
