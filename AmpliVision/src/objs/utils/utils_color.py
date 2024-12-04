import numpy as np
import cv2 as cv
from statistics import mode

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def get_rgb_avg_of_contour(square, contour: np.ndarray, debug: bool = False) -> list[int]:
    "rgb avg of any shapped contour"

    # copy the image
    image = square.get_test_area_img().copy()

    # turn contour into mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, (255), -1)

    # get the pixels inside the contour
    pixels_inside = image[mask == 255]

    #dp = cv.bitwise_and(image, image, mask=mask)
    #cv.imshow('utils_color/get_rgb_avg_of_contour', dp)
    #cv.waitKey(0)
    cv.destroyAllWindows()

    # Calculate the mode RGB values
    mode_rgb = [ mode(pixels_inside[:, 0]), mode(pixels_inside[:, 1]), mode(pixels_inside[:, 2]) ]

    """
    # Plot the histogram of R, G, and B channels in subplots
    fig, axs = plt.subplots(3)
    fig.suptitle('RGB Histogram')

    colors = ['red', 'green', 'blue']
    curve_mean = []
    "TODO: split calculation and plotting into two functions"
    for i in range(3):
        # Compute the histogram data
        hist, bin_edges = np.histogram(pixels_inside[:, i], bins=256, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit the Lorentzian function to the histogram data
        try:
            popt, _ = curve_fit(
                lorentzian, 
                bin_centers, 
                hist, 
                p0=[
                    max(hist), 
                    np.mean(pixels_inside[:, i]), 
                    np.std(pixels_inside[:, i])
                ],
                bounds = (0, 255)
            ) 
        except RuntimeError as e:
            print(f"Error: {e}. Using mode_rgb")
            popt = mode_rgb
        mean = popt[1]
        curve_mean.append(mean)

        # Plot the histogram
        axs[i].hist(pixels_inside[:, i], bins=256, color=colors[i], alpha=0.6, label='Histogram')
        axs[i].set_title(f'Channel {colors[i]}')
        axs[i].set_xlim([0, 256])
        
        # Plot the Lorentzian fitted curve
        fitted_curve = lorentzian(bin_centers, *popt)
        axs[i].plot(bin_centers, fitted_curve, color='black', linewidth=2, label='Fitted Lorentzian curve')
        
        # Plot the mode with a different color
        axs[i].axvline(mode_rgb[i], color='k', linestyle='dashed', linewidth=2, label='Mode')
        axs[i].axvline(mean, color='gold', linestyle='dashed', linewidth=2, label='Curve Mean')

        # Add a legend
        axs[i].legend()
        
        print("\n"*3)
        # print plot x, y values
        print(f"mode_rgb[{i}] = {mode_rgb[i]}")
        print(f"curve_mean[{i}] = {mean}")

        print(f"X values: {bin_centers}")
        print(f"Curve Y values: {fitted_curve}")
        print(f"Histogram Y values: {hist}")

        print("\n")
    print("\n"*3)
    plt.show()
    plt.close()
    
    
    

    # Remove NaN values
    mode_rgb = [round(x) for x in curve_mean]
    mode_rgb = np.nan_to_num(mode_rgb)
    #"""
    

    # display the mask drawn on the image
    if debug:
        square.get_test_area_img().copy()
        cv.drawContours(image, [contour], -1, (0, 255, 0), 1)
        cv.imshow('utils_color/get_rgb_ag_of_contour', image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return [round(x) for x in mode_rgb]

def get_rgb_avg_of_circle_contour(square, contour: np.ndarray, debug: bool = False) -> list[int]:
    """
    ### Get RGB average of contour
    ---------------
    Function that gets the average RGB of a contour in the image.

    #### Args:
    * contour: Contour of the object in the image.
    * corner: Corner of the square the contour is in.

    #### Returns:
    * avg_color: Average RGB color of the contour.
    """

    # copy the image
    image = square.img.copy()

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB format

    if debug == 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    (x, y), radius = cv.minEnclosingCircle(contour)

    center = (int(x), int(y))
    radius = max(int(radius) - int(square.PLUS_MINUS/2.5), 3)

    # get the pixels inside the minEnclosingCircle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.circle(mask, center, radius, (255), -1)
    pixels_inside = image[mask == 255]

    if debug == 2:
        pixels_inside = square.img[mask == 255]

    # Calculate the average RGB values
    average_rgb = np.mean(pixels_inside, axis=0)

    # Remove NaN values
    average_rgb = np.nan_to_num(average_rgb)

    return [round(x) for x in average_rgb]


def get_pins_rgb(square) -> tuple[list[int], list[int]]:
    """ 
    gets the average RGB of the pins in the square.
    """

    pins_rgb = []
    corner = []

    # for each pin in the square get the average RGB value of the pin and its corner
    for pins in square.pins:
        corner.append(square.which_corner_is_contour_in(pins))
        pins_rgb.append(get_rgb_avg_of_circle_contour(square, pins, corner))

    return pins_rgb, corner  # tr, tl, br, bl corners


def set_rgb_sequence_clockwise(square, pins_rgb: list[int], corner_key: list[int]) -> None:
    """sets the rgb sequence of the square in clockwise order starting from top-left."""

    sequence = []
    for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        sequence.append(pins_rgb[corner_key.index(key)]
                        if key in corner_key else (0, 0, 0))

    square.rgb_sequence = sequence


# Define the Lorentzian function
def lorentzian(x, a, x0, b):
    return (a / np.pi) * (b / ((x - x0)**2 + b**2))
