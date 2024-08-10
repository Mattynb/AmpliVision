import cv2 as cv
from datetime import datetime
from .strip_section import StripSection
from ..image.processors.image_processor import ColorContourExtractor


class TestAnalyzer:
    "This class is responsible for getting and analyzing test results a.k.a phase B"

    def __init__(self, block):

        self.block = block

        # look only at the inner test square:
        self.test_square_img = block.get_test_area_img()

        # square used in csv export
        self.grid_index = block.index
        self.block_type = block.get_block_type()

        self.strip_sections = {
            "bkg": StripSection(self.test_square_img, 'bkg', block.rotation),
            "spot1": StripSection(self.test_square_img, 'spot1', block.rotation),
            "spot2": StripSection(self.test_square_img, 'spot2', block.rotation)
        }

    def analyze_test_result(self, display: bool = False):  # should I name it main?
        "gets test results from a block, analyses them, and export them to csv"

        # find the positive spots with hsv mask
        # need to think about cases where mask for example return one pixel.
        #   do you check for minimum contour size? do you only look for it manually? food for thought

        if display:
            print("rotation: ", self.block.rotation)

        blur = cv.GaussianBlur(self.test_square_img, (3, 3), 0)
        rgb_spots = ColorContourExtractor.process_image(
            blur, hsv_lower=[0, 40, 20], display=display)

        if display:
            copy = self.test_square_img.copy()
            cv.drawContours(copy, rgb_spots, -1, (0, 255, 0), 1)

        self.add_positives_to_sections(rgb_spots, display=display)

        # find the negative spots "manually" through ratios
        self.add_negatives_to_sections(display=display)

        # get background color noise so we can remove it from other sections
        self.strip_sections['bkg'].set_total_avg_rgb()
        bkg_rgb_avg = self.strip_sections['bkg'].total_avg_rgb

        # remove background noise from other sections
        corrected_rgbs = []
        for section in self.strip_sections.values():
            if section.strip_type != 'bkg':

                if display:
                    print("correcting: ", section.strip_type)

                corrected_rgbs.append(section.subtract_bkg(bkg_rgb_avg))

        if display:
            print("\n")

        # validate results to catch any potential errors in the test
        "TODO: adapt validate_results to work with the new strip configuration"
        # self.validate_results()

        for section in self.strip_sections.values():
            section.set_total_avg_rgb()
            # print("total avg rgb in ", section.strip_type, " is: ", section.total_avg_rgb)

        # export results to csv
        return self.create_csv_row(corrected_rgbs)

    def add_positives_to_sections(self, rgb_spots, display: int = 0) -> None:
        "used to add positive result spots to appropriate strip section"

        # adds each spot to its strip section
        for spot in rgb_spots:
            for section in self.strip_sections.values():
                if section.bounds_contour(spot):

                    if display:
                        print("auto added spot to: ", section.strip_type)
                    section.add_spot(self.block, spot, True, debug=display)
                    # break # only adds to one section

    def add_negatives_to_sections(self, display: int = 0) -> None:
        "used to find negative result spots to appropriate strip section"
        for type, section in zip(self.strip_sections.keys(), self.strip_sections.values()):

            if len(section.spots) == 0:
                if display:
                    print("man added negative spot to: ", type)
                section.set_spots_manually(self.block, debug=display)

    def validate_results(self) -> None:
        "deals with test result potential positive, negative, false positive, error scenarios"

        results = self.get_section_results()

        # 1 test is properly positive (bkg, test, and spot2 line rgbs are > threshold)
        if results[1] & results[2]:
            print("Test worked properly and result is positive")

        # 2 test is properly negative (spot2 line rgb is > threshold)
        elif (not results[1]) & results[2]:
            print("Test worked properly and result is negative")

        # 3 spot2 error (bkg, and maybe test line rgbs are > threshold)
        else:
            print("Test may have not worked properly")

    def get_section_results(self) -> list[bool]:
        "returs a list of booleans representing the result (positive or negative) of each section bkg, test, spot2"
        results = []  # bkg, test, spot2
        for strip in self.strip_sections.values():
            strip_result = False

            # display
            # strip.print_spots()

            for spot in strip.spots:
                if spot["positive"] == True:
                    strip_result = True
                    break
            results.append(strip_result)

        print("\n")

        return results

    def create_csv_row(self, corrected_rgbs: list[list]) -> str:
        """
        writes the test results to csv file row in format:\n
        date, time, grid_index, block_type, bkg_r, bkg_g, bkg_b, test_r, test_g, test_b, cntrl_r, cntrl_g, cntrl_b"""

        # get current date and time
        now = datetime.now()

        # format date and time
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")

        # setting all rgb values to None
        bkg_r = " None"
        bkg_g, bkg_b = bkg_r, bkg_r
        spot1_r, spot1_g, spot1_b = bkg_r, bkg_r, bkg_r
        spot2_r, spot2_g, spot2_b = bkg_r, bkg_r, bkg_r

        # get rgb values of each section
        if self.strip_sections['bkg'].total_avg_rgb != None:
            bkg_b, bkg_g, bkg_r = self.strip_sections['bkg'].total_avg_rgb
            # print("bkg rgb: ", bkg_r, bkg_g, bkg_b)

        if self.strip_sections['spot1'].total_avg_rgb != None:
            spot1_b, spot1_g, spot1_r = self.strip_sections['spot1'].total_avg_rgb
            # print("spot1 rgb: ", test_r, test_g, test_b)

        if self.strip_sections["spot2"].total_avg_rgb != None:
            spot2_b, spot2_g, spot2_r = self.strip_sections['spot2'].total_avg_rgb
            # print("spot2 rgb: ", cntrl_r, cntrl_g, cntrl_b)

        spot1_corr_b, spot1_corr_g, spot1_corr_r = corrected_rgbs[0]
        spot2_corr_b, spot2_corr_g, spot2_corr_r = corrected_rgbs[1]

        # create data to be written to csv
        data = [
            date, time, self.grid_index, self.block_type,
            spot1_r, spot1_g, spot1_b,
            spot2_r, spot2_g, spot2_b,
            bkg_r, bkg_g, bkg_b,
            spot1_corr_r, spot1_corr_g, spot1_corr_b,
            spot2_corr_r, spot2_corr_g, spot2_corr_b
        ]

        return data

    def paint_spots(self, rgb_spot_results: dict[list]):
        """ colors the spots in the image accourding to the results 

        Args:
            rgb_spot_results (dict[list]): dictionary containing the rgb values of the spots in the image
            format: {r: [mean1, std1, mean2, std2], g: [mean1, std1, mean2, std2], b: [mean1, std1, mean2, std2]}
        """
        from numpy import random

        for section in self.strip_sections.values():
            if section == 'bkg':
                continue

            if section == 'spot1':
                i = 0

            if section == 'spot2':
                i = 2

            r = random.normal(
                rgb_spot_results['r'][i], rgb_spot_results['r'][i+1])
            g = random.normal(
                rgb_spot_results['g'][i], rgb_spot_results['g'][i+1])
            b = random.normal(
                rgb_spot_results['b'][i], rgb_spot_results['b'][i+1])
            rgb = (r, g, b)

            return section.paint_spot(self.test_square_img, rgb, True)


"""TODO: Add a stripSection "middleman" class to take care of assigning which spots go to which section, etc"""
