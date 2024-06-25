from operator import ge
import numpy as np

class DataGenerator:
    def __init__(self, combined_fingerprints:dict):
        self.combined_fingerprints=combined_fingerprints

    def generate_n_points_by_blocktype(self, blocktype:str, n_points:int)->list[list[list[float]]]:
        """ 
        generates n random spot1&2 points based on the blocktype's spot1&2 normal distribution


        Each iteration generates points for r, g, or b:
        - spot1_points = [r1, r1, r1, r1, r1, r1, ... n_points]
        - spot2_points = [r2, r2, r2, r2, r2, r2, ... n_points]
        - ...

        [
            [spot1_r [...],  spot2_r [...]]
            [spot1_g [...],  spot2_g [...]]
            [spot1_b [...],  spot2_b [...]]
        ]
        """

        generated = []
        for rgb in ['r','g', 'b']:
            rgb_mean1, rgb_std1, rgb_mean2, rgb_std2 = self.combined_fingerprints[blocktype][rgb] 

            spot1_points = [int(x) if x > 0 else 0 for x in np.random.normal(rgb_mean1, rgb_std1, n_points)]
            spot2_points = [int(x) if x > 0 else 0 for x in np.random.normal(rgb_mean2, rgb_std2, n_points)]

            #print(f"spot1_{rgb}: ", spot1_points,f" \t|\tspot2_{rgb}: ", spot2_points)

            generated.append([spot1_points, spot2_points])

        return generated
    
    def separate_points(self, points:list):
        """
        Takes:
        [
            [spot1_r [...],  spot2_r [...]]
            [spot1_g [...],  spot2_g [...]]
            [spot1_b [...],  spot2_b [...]]
        ]
        
        Transforms to:
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

            # [r,g, or b][spot 1 or 2][row i]
            r1 = points[0][0][i]
            g1 = points[1][0][i]
            b1 = points[2][0][i]

            r2 = points[0][1][i]
            g2 = points[1][1][i]
            b2 = points[2][1][i]

            separated.append([r1,g1,b1,r2,g2,b2])

        return separated



    def generate_n_points(self, n_points)-> dict[list[list[float]]]:
        """ generates n random points for each block type

        Returns:
        {
            "type1": [[r1, g1, b1, r2, g2, b2], ... ],
            "type2": [[r1, g1, b1, r2, g2, b2], ... ],    
            ...
        }
        """
        generated_points = {}
        for blocktype in self.combined_fingerprints:
            pts = self.generate_n_points_by_blocktype(blocktype, n_points)
            pts = self.separate_points(pts)
            generated_points[blocktype] = pts
        return generated_points



if __name__ == '__main__':
    combined=  {'Test1': {'r': [-0.25, 3.031088913245535, 18.5, 10.0], 'g': [0.75, 2.384848003542364, 90.375, 10.0], 'b': [-0.125, 3.822967336793113, 56.375, 10.0]}, 'Test2': {'r': [29.75, 10.0, 69.5, 10.0], 'g': [61.25, 10.0, 94.75, 10.0], 'b': [57.125, 10.0, 89.875, 10.0]}}

    dg = DataGenerator(combined)
    print(dg.generate_n_points(10))