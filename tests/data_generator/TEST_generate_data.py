import unittest
from src.data_generator.src.data_extractor import DataExtractor

class TestDataExtractor(unittest.TestCase):

    def setUp(self):
        self.sample_type = 'sample_type'
        self.results_folder_path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024'
        self.data_extractor = DataExtractor(self.sample_type, self.results_folder_path)

    def test_extract_fingerprints_across_files(self):
        # Prepare test data
        path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024\DENV_IMG_1.JPEG_results_06-06-2024_(16-27-23).csv'

        ...

    def test_append_fingerprints(self):
        # Prepare test data
        fingerprints1 = [{'block_type1': {'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}}, {'block_type1': {'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}}]
        fingerprints2 = [{'block_type1': {'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}, 'block_type2': {'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}}]

        # Call the function
        combined_fingerprint1 = self.data_extractor.append_fingerprints(fingerprints1)
        combined_fingerprint2 = self.data_extractor.append_fingerprints(fingerprints2)

        # Assertion
        # 1 block type
        expected_combined_fingerprint1 = {'block_type1': [{'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}, {'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}]}
        self.assertEqual(combined_fingerprint1, expected_combined_fingerprint1, 'Same block type')
        
        # 2 different block types
        expected_combined_fingerprint2 = {'block_type1': [{'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}], 'block_type2': [{'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}]}
        self.assertEqual(combined_fingerprint2, expected_combined_fingerprint2, 'Different block types')

    
    def test_combine_fingerprints(self):
        # Prepare test data
        appended_fingerprints = {
            'block_type1': [
                {
                    'r': [1, 2, 3, 4],
                    'g': [5, 6, 7, 8],
                    'b': [9, 10, 11, 12]
                },
                {
                    'r': [13, 14, 15, 16],
                    'g': [17, 18, 19, 20],
                    'b': [21, 22, 23, 24]
                }
            ],
            'block_type2': [
                {
                    'r': [25, 26, 27, 28],
                    'g': [29, 30, 31, 32],
                    'b': [33, 34, 35, 36]
                },
                {
                    'r': [37, 38, 39, 40],
                    'g': [41, 42, 43, 44],
                    'b': [45, 46, 47, 48]
                }
            ]
        }

        # Call the function
        combined_fingerprints = self.data_extractor.combine_fingerprints(appended_fingerprints)

        # Assertion
        expected_combined_fingerprints = {
            'block_type1': {
                'r': [7, 8, 9, 10],
                'g': [11, 12, 13, 14],
                'b': [15, 16, 17, 18]
            },
            'block_type2': {
                'r': [31, 32, 33, 34],
                'g': [35, 36, 37, 38],
                'b': [39, 40, 41, 42]
            }
        }
        self.assertEqual(combined_fingerprints, expected_combined_fingerprints, 'Combined fingerprints')


    def test_get_mean_and_std_of_spots(self):
        spot1_corr_rgbs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        spot2_corr_rgbs = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
        
        mean_and_std = self.data_extractor.get_mean_and_std_of_spots(spot1_corr_rgbs, spot2_corr_rgbs)
        
        expected_mean_and_std = {
            'r': [4, 2.449489742783178, 13, 2.449489742783178],
            'g': [5, 2.449489742783178, 14, 2.449489742783178],
            'b': [6, 2.449489742783178, 15, 2.449489742783178]
        }
        self.assertEqual(mean_and_std, expected_mean_and_std, 'Mean and standard deviation of spots')

    def test_get_mean_and_std_of_results(self):
        image_results = [
            {
                'r': [1, 2, 3, 4],
                'g': [5, 6, 7, 8],
                'b': [9, 10, 11, 12]
            },
            {
                'r': [13, 14, 15, 16],
                'g': [17, 18, 19, 20],
                'b': [21, 22, 23, 24]
            }
        ]
        rgb = 'r'
        mean_and_std = self.data_extractor.get_mean_and_std_of_results(image_results, rgb)
        expected_mean_and_std = [7, 8, 9, 10]
        self.assertEqual(mean_and_std, expected_mean_and_std, 'Mean and standard deviation calculation')

    def test_extract_corr_rgbs(self):
        data = [
            "date       ,time     ,grid_index ,block_type    ,spot1_r ,spot1_g ,spot1_b ,spot2_r ,spot2_g ,spot2_b ,bkg_r ,bkg_g ,bkg_b ,spot1_corr_r ,spot1_corr_g ,spot1_corr_b ,spot2_corr_r ,spot2_corr_g ,spot2_corr_b",
            "06/06/2024 ,16:28:13 ,\"(3, 4)\"   ,type1 ,  217.0 ,  213.0 ,  213.0 ,  217.0 ,  214.0 ,  216.0 ,214.0 ,212.0 ,211.0 ,1         ,2         ,3         ,4         ,5         ,6",
            "06/06/2024 ,16:28:33 ,\"(4, 4)\"   ,type2   ,  126.0 ,  119.0 ,  163.0 ,   90.0 ,   88.0 ,  121.0 ,209.0 ,205.0 ,207.0 ,7         ,8.0         ,9         ,10.0         ,11.0        ,12.0"  
        ]
        rgbs_by_type = self.data_extractor.extract_corr_rgbs(data)
        
        expected_rgbs_by_type = {
            'type1': [[[1,2,3], [4,5,6]]],
            'type2': [[[7,8,9], [10,11,12]]]
        }
        self.assertEqual(rgbs_by_type, expected_rgbs_by_type, 'Extract corr rgbs from data')


if __name__ == '__main__':
    unittest.main()