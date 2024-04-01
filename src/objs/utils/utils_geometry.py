
from math import sqrt
import cv2 as cv
import numpy as np
from cv2.typing import MatLike

# Euclidian Distance
def distance(p1:float, p2:float):
    """
    p1 = (x1,y1)
    p2 = (x2,y2)
    """
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# checks if a combination of points are arranged in the shape of a square 
def is_arranged_as_square(points:list[tuple], img, SQUARE_LENGTH:int, flag=0):
    """
    checks if a combination of points are arranged in the shape of a square
    ----------
    points= combination of 4 points (x,y)
    """
    
    # Convert tuples to lists
    points = [list(point) for point in points]
    
    # flag is used to check recursion depth.
    # recursion is necessary in the case 
    # where points are in the order of (x0,y0), (x1,y1), (x3,y3), (x2,y2)
    if flag:
        points[2], points[3] = points[3], points[2]

    # Calculate distances between each pair of points and diagonals
    dists = calculate_distances(points)    

    # Check if the points form a square
    if is_square(points, dists, SQUARE_LENGTH):
        draw_square_on_image(points, img)
        return True
    
    if not flag:
        return is_arranged_as_square(points, img, SQUARE_LENGTH, 1)

    return False

def calculate_distances(points: list[tuple]) -> list[float]:
        """
        Calculate distances between each pair of points and diagonals.
        ----------
        points: list of 4 points (x,y)
        """
        # Assuming points is a list of four (x, y) tuples
        # Calculate distances between each pair of points
        dists = []
        for i in range(3):
            dists.append(distance(points[i], points[i + 1]))
        dists.append(distance(points[3], points[0]))

        # corners
        dists.append(distance(points[0], points[2]))
        dists.append(distance(points[1], points[3]))

        return dists

def is_square(points: list[tuple], dists: list[float], SQUARE_LENGTH: int):
    """
    Checks if a set of points forms a square.
    
    ----------
    points: list of 4 points (x,y)
    dists: list of distances between each pair of points
    SQUARE_LENGTH: expected side length of the square

    Arranged as: 
    \n
    3---2

    0---1 
    """
    return (
        # it is around the size of a square
        all(dist < SQUARE_LENGTH for dist in dists)

        # x0 == x3
        and np.isclose(points[0][0], points[3][0], atol=0.1, rtol=0.1) 
        
        # y0 == y1
        and np.isclose(points[0][1], points[1][1], atol=0.1, rtol=0.1) 
        
        # x1 == x2
        and np.isclose(points[1][0], points[2][0], atol=0.1, rtol=0.1) 
        
        # y2 == y3
        and np.isclose(points[2][1], points[3][1], atol=0.1, rtol=0.1)

        # diagonals are equal
        and np.isclose(dists[5], dists[4], atol=0.1, rtol=0.1)
        
        # sides are equal
        and np.isclose(dists[0], dists[1], atol=0.1, rtol=0.1)
        and np.isclose(dists[1], dists[2], atol=0.1, rtol=0.1)
        and np.isclose(dists[2], dists[3], atol=0.1, rtol=0.1)
    )


def draw_square_on_image(points: list[tuple], img):
    """
    Draws a square on the image using the provided points.
    ----------
    points: list of 4 points (x,y)
    img: the image on which to draw the square
    """
    copy = img.copy()
    for i in range(len(points)):
        cv.circle(copy, points[i], 5, (0, 0, 255), -1)
        cv.line(copy, points[i], points[(i+1)%4], (0, 0, 255), 2)
        cv.putText(copy, f"{i}", points[i], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('image', copy)


# Finds center point of contour 
def find_center_of_contour(contour: MatLike):   
    """
    Finds Center point of a single contour
    ---------
    contour: single contour

    """
    M = cv.moments(contour)  

    # avoiding division by zero
    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    
        return (x, y)
    else:
        
        return None

# Finds center point of points
def find_center_of_points(points: list[tuple]):
    """
    Finds Center point of a list of points
    ---------
    points: list of points

    """
    x = 0
    y = 0

    for point in points:
        x += point[0]
        y += point[1]

    return (x//len(points), y//len(points))

# Not being used as of 03/18/2024
def contour_is_circular(contour: MatLike):
    """
    ### Contour is circular
    ---------------
    Function that checks if a contour is circular.
    
    #### Args:
    * contour: Contour of the object in the image.
    """

    # Approximate the contour
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    # Check the circularity
    check_1 = False
    if 0.6 < circularity < 1.4:
        # This contour is close to a circle
        check_1 = True

    # Fit a bounding rectangle and check the aspect ratio
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    check_2 = False
    if 0.6 < aspect_ratio < 1.4:
        # The contour is close to being contained in a square
        check_2 = True

    # Minimum enclosing circle
    (x, y), radius = cv.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    check_3 = False
    if 0.6 < (area / circle_area) < 1.4:
        # The area of the contour is close to that of the enclosing circle
        check_3 = True
    
    if check_1 and check_2 and check_3:
        return True

    return False