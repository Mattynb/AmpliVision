import unittest
from unittest.mock import patch
from src.data_generator.data_generator import DataGenerator

class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.sample_type = 'sample_type'
        self.results_folder_path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024'
        self.data_generator = DataGenerator(self.sample_type, self.results_folder_path)

    def test_extract_fingerprints_across_files(self):
        # Prepare test data
        path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024\DENV_IMG_1.JPEG_results_06-06-2024_(16-27-23).csv'

        ...

    def test_append_fingerprints(self):
        # Prepare test data
        fingerprints1 = [{'block_type1': {'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}}, {'block_type1': {'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}}]
        fingerprints2 = [{'block_type1': {'r': [1, 2, 3, 4], 'g': [5, 6, 7, 8], 'b': [9, 10, 11, 12]}, 'block_type2': {'r': [13, 14, 15, 16], 'g': [17, 18, 19, 20], 'b': [21, 22, 23, 24]}}]

        # Call the function
        combined_fingerprint1 = self.data_generator.append_fingerprints(fingerprints1)
        combined_fingerprint2 = self.data_generator.append_fingerprints(fingerprints2)

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
        combined_fingerprints = self.data_generator.combine_fingerprints(appended_fingerprints)

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


    def test_get_mean_and_std(self):
        # Prepare test data
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

        # Call the function
        mean_and_std = self.data_generator.get_mean_and_std(image_results, rgb)

        # Assertion
        expected_mean_and_std = [7, 8, 9, 10]
        self.assertEqual(mean_and_std, expected_mean_and_std, 'Mean and standard deviation calculation')

if __name__ == '__main__':
    unittest.main()