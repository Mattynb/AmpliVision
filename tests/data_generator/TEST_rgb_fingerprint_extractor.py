import csv
import unittest
from io import StringIO
from src.data_generator import get_corr_rgbs, format_line, get_mean_and_std, limit_std


class TestFunctions(unittest.TestCase):

    def test_get_corr_rgbs(self):
        # Prepare test data
        data = [
            "date,time,grid_index,block_type,spot1_r,spot1_g,spot1_b,spot2_r,spot2_g,spot2_b,bkg_r,bkg_g,bkg_b,spot1_corr_r,spot1_corr_g,spot1_corr_b,spot2_corr_r,spot2_corr_g,spot2_corr_b\n",
            "06/06/2024,16:28:13,\"(3, 4)\",type1,217.0,213.0,213.0,217.0,214.0,216.0,214.0,212.0,211.0,-2.0,-1.0,-3.0,-5.0,-2.0,-3.0\n",
            "06/06/2024,16:28:33,\"(4, 4)\",type2,126.0,119.0,163.0,90.0,88.0,121.0,209.0,205.0,207.0,44.0,86.0,83.0,86.0,117.0,119.0\n"
        ]

        # Expected result
        expected_rgbs = {
            'type1': [[[-2.0,-1.0,-3.0],[-5.0,-2.0,-3.0]]],
            'type2': [[[44.0,86.0,83.0],[86.0,117.0,119.0]]]
        }

        # Call the function
        rgbs_by_type = get_corr_rgbs(data)

        # Assertion
        self.assertEqual(rgbs_by_type.keys(), expected_rgbs.keys())
        for block_type in rgbs_by_type:
            self.assertEqual(rgbs_by_type[block_type], expected_rgbs[block_type])

    def test_format_line(self):
        # Prepare test data
        line = "06/06/2024,16:28:33,\"(4, 4)\",type2,126.0,119.0,163.0,90.0,88.0,121.0,209.0,205.0,207.0,44.0,86.0,83.0,86.0,117.0,119.0\n"
        # split the line by comma, skipping commas inside quotes
        line = csv.reader(StringIO(line), delimiter=',', quotechar='"').__next__()
        
        # Expected result
        expected_result = [44.0,86.0,83.0]

        # Call the function
        formatted_line = format_line(line, 0)

        # Assertion
        self.assertEqual(formatted_line, expected_result)

    def test_get_mean_and_std(self):
        # Prepare test data
        spot1_corr_rgbs = [[100.0, 50.0, 25.0], [200.0, 100.0, 50.0]]
        spot2_corr_rgbs = [[200.0, 100.0, 50.0], [150.0, 75.0, 37.0]]

        # Expected result (simplified for demonstration)
        expected_result = {
            'r': [150.0, 50.0, 175.0, 25.0], # mean_spot1, std_spot1, mean_spot2, std_spot2
            'g': [75.0, 25.0, 87.5, 12.5],
            'b': [37.5, 12.5, 43.5, 6.5]
        }

        # Call the function
        fingerprint = get_mean_and_std(spot1_corr_rgbs, spot2_corr_rgbs)

        # Assertion
        for color in expected_result:
            self.assertEqual(fingerprint[color], expected_result[color], msg=f"Color {color} failed")

    def test_limit_std(self):
        # Prepare test data
        fingerprint = {
            'r': [150.0, 50.0, 175.0, 30.0],
            'g': [75.0, 25.0, 87.5, 20.0],
            'b': [37.0, 12.5, 31.0, 10.0]
        }
        limit = 15

        # Expected result
        expected_result = {
            'r': [150.0, 15.0, 175.0, 15.0],
            'g': [75.0, 15.0, 87.5, 15.0],
            'b': [37.0, 12.5, 31.0, 10.0]
        }

        # Call the function
        limited_fingerprint = limit_std(fingerprint, limit)

        # Assertion
        self.assertEqual(limited_fingerprint, expected_result)
        

if __name__ == '__main__':
    # Run the tests showing pass and fail results
    unittest.main()
