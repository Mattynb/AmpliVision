""" Unit tests for DataExtractor """

import unittest
from unittest.mock import patch, mock_open

from src.data_extractor import DataExtractor


class TestDataExtractor(unittest.TestCase):
    """ Class that holds the unit tests for DataExtractor"""

    def setUp(self):
        self.sample_type = 'IMG'
        self.extract_from = 'tests/test_data/'
        self.data_extractor = DataExtractor(
            self.sample_type, self.extract_from)

    @patch('src.data_extractor.glob')
    def test_load_csv_files(self, mock_glob):
        mock_glob.return_value = ['IMG_6280.csv', 'IMG_6283.csv']
        csv_files = self.data_extractor.load_csv_files()
        self.assertEqual(csv_files, ['IMG_6280.csv', 'IMG_6283.csv'])
        mock_glob.assert_called_once_with('tests/test_data/IMG*.csv')

    @patch('src.data_extractor.glob')
    def test_load_csv_files_with_custom_path(self, mock_glob):
        mock_glob.return_value = ['IMG_6280.csv', 'IMG_6283.csv']
        csv_files = self.data_extractor.load_csv_files(self.extract_from)
        self.assertEqual(csv_files, ['IMG_6280.csv', 'IMG_6283.csv'])
        mock_glob.assert_called_once_with('tests/test_data/IMG*.csv')

    def test_separate_spots(self):
        blocks = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        spot1_corr_rgbs, spot2_corr_rgbs = self.data_extractor.separate_spots(
            blocks)
        self.assertEqual(spot1_corr_rgbs, [[1, 2, 3], [7, 8, 9]])
        self.assertEqual(spot2_corr_rgbs, [[4, 5, 6], [10, 11, 12]])

    def test_append_spots(self):
        rgbs_by_type = [
            {'type1': [[[1, 2, 3], [4, 5, 6]]]},
            {'type1': [[[7, 8, 9], [10, 11, 12]]]}
        ]
        expected_output = {
            'type1': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        }
        appended_spots = self.data_extractor.append_spots(rgbs_by_type)
        self.assertEqual(appended_spots, expected_output)

    read_data = """date,time,grid_index,block_type,
    spot1_r,spot1_g,spot1_b,
    spot2_r,spot2_g,spot2_b,
    bkg_r,bkg_g,bkg_b,
    spot1_corr_r,spot1_corr_g,spot1_corr_b,
    spot2_corr_r,spot2_corr_g,spot2_corr_b""".replace('\n', '').replace(' ', '') + """
    2021-01-01,12:00,1,type1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150""".replace(' ', '')

    @patch('builtins.open', new_callable=mock_open, read_data=read_data)
    def test_extract_spots_from_single_csv(self, mock_file):
        path = 'tests/test_data/IMG_6280.csv'
        expected_output = {'type1': [[[100, 110, 120], [130, 140, 150]]]}
        result = self.data_extractor.extract_spots_from_single_csv(path)
        self.assertEqual(result, expected_output)
        mock_file.assert_called_once_with(path, 'r', encoding='UTF-8')

    def test_get_mean_and_std_of_spots(self):
        spot1_corr_rgbs = [[1, 2, 3], [4, 5, 6]]
        spot2_corr_rgbs = [[7, 8, 9], [10, 11, 12]]
        expected_output = {
            'r': [2.5, 1.5, 8.5, 1.5],
            'g': [3.5, 1.5, 9.5, 1.5],
            'b': [4.5, 1.5, 10.5, 1.5]
        }
        result = self.data_extractor.get_mean_and_std_of_spots(
            spot1_corr_rgbs, spot2_corr_rgbs)
        self.assertEqual(result, expected_output)

    @patch('src.data_extractor.DataExtractor.load_csv_files')
    @patch('src.data_extractor.DataExtractor.extract_spots_from_multiple_csv')
    @patch('src.data_extractor.DataExtractor.convert_spots_to_fingerprints')
    def test_extract(self, mock_convert, mock_extract_spots, mock_load_csv):
        mock_load_csv.return_value = ['file1.csv']
        mock_extract_spots.return_value = [{'type1': [[[1, 2, 3], [4, 5, 6]]]}]
        mock_convert.return_value = {'type1': [1, 2, 3]}

        result = self.data_extractor.extract()

        self.assertEqual(result, {'type1': [1, 2, 3]})
        mock_load_csv.assert_called_once()
        mock_extract_spots.assert_called_once_with(['file1.csv'])
        mock_convert.assert_called_once_with(
            [{'type1': [[[1, 2, 3], [4, 5, 6]]]}])


if __name__ == '__main__':
    unittest.main()
