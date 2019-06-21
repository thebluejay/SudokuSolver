#!/usr/bin/python


# SUDOKU SOLVER
# Author: KoffeinFlummi
# Requires:
#   - numpy
#   - OpenCV2
#   - An installation of tesseract

# DESCRIPTION
# Tries to extract a sudoku from an image and solve it.
# While it isn't always able to extract every digit from the
# given sudoku, that is usually enough to be able to solve it.

# USAGE
# Called from the command line with the filename of the image
# containing the sudoku as an argument, like so:
# $ python sudokusolver.py mysuperhardsudoku.jpg

# GETTING THE BEST RESULTS
# The script needs to be able to detect the shape of the sudoku,
# so please make sure that there's nothing but the sudoku in the
# picture and that the paper doesn't have any bends or creases
# in it that might make it harder to detect square shapes.


import sys
import os
import subprocess
import copy
import numpy as np

import cv2

DEBUG = False

def find_free_filename(ext):
    """ Finds a free filename to store a temporary file under. """
    parent = os.path.realpath(__file__)
    for i in range(100):
        filename = "temp"+str(i)+"."+ext
        if not os.path.exists(os.path.join(parent, filename)):
            return filename
    return filename

def debug_image(img, name):
    """ Shows an image for debugging. """
    if not DEBUG:
        return

    cv2.namedWindow(name)
    cv2.imshow(name, img)
    #imwrite(name+".jpg", img) # uncomment if you want to save the individual steps
    cv2.waitKey()

def print_sudoku(sudoku):
    """ Prints a sudoku thingy nicely. """
    print("+-------+-------+-------+")
    for row in range(9):
        print("| {} {} {} | {} {} {} | {} {} {} |".format(*sudoku[row]).replace("0", " "))
        if (row+1) % 3 == 0:
            print("+-------+-------+-------+")

def project_image(img):
    """ Compensates for perspective by finding the outline and making the sudoku a square again. """
    # Grayscale image for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 200)

    debug_image(canny, "edgedetected")

    # Detect contours
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours for things that might be squares
    squares = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(contour) == 4 and cv2.isContourConvex(contour):
            squares.append(contour)

    # Find the biggest one.
    squares = [sorted(squares, key=lambda x: cv2.contourArea(x))[-1]]
    squares[0] = squares[0].reshape(-1, 2)

    imgcontours = img
    cv2.drawContours(imgcontours, squares, -1, (0, 0, 255))
    debug_image(imgcontours, "squares")

    # Arrange the border points of the contour we found so that they match points_new.
    points_original = sorted(squares[0], key=lambda x: x[0])
    points_original[0:2] = sorted(points_original[0:2], key=lambda x: x[1])
    points_original[2:4] = sorted(points_original[2:4], key=lambda x: x[1])
    points_original = np.float32(points_original)
    points_new = np.float32([[0, 0], [0, 450], [450, 0], [450, 450]])

    # Warp the image to be a square.
    pers_trans = cv2.getPerspectiveTransform(points_original, points_new)
    fixed_image = cv2.warpPerspective(img, pers_trans, (450, 450))

    debug_image(fixed_image, "perspectivefix")

    return fixed_image

def extract_sudoku(img):
    """ Extracts the actual numbers from the image using tesseract. """
    sudoku = []
    for row in range(9):
        sudoku_temp = []
        for col in range(9):
            border = 5 # how much to cut off the edges to eliminate any of the lines between the cells
            subimg = img[row*50+border:(row+1)*50-border, col*50+border:(col+1)*50-border]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2RGB)
            ret, thresh = cv2.threshold(subimg, 127, 255, cv2.THRESH_BINARY) # black-and-white for most contrast

            tesinput = find_free_filename("jpg")
            tesoutput = find_free_filename("txt")
            cv2.imwrite(tesinput, thresh)

            try:
                subprocess.check_output("tesseract "+tesinput+" "+tesoutput[:-4]+" --psm 10 quiet", shell=True)
                digit = int(open(tesoutput, "r").read())
            except:
                digit = 0

            # try:
            #     os.remove(tesinput)
            #     os.remove(tesoutput)
            # except:
            #     print("Failed to delete temp files. Probably some bullshit with permissions.")
            #     sys.exit(1)

            sudoku_temp.append(digit)
            sys.stdout.write("\r"+"Extracting data ... "+str(int((row*9+col+1)/81*100)).rjust(3)+"%")
            sys.stdout.flush()
        sudoku.append(sudoku_temp)

    return sudoku

def is_valid_solution(sudoku):
    """ Checks if the given sudoku is a valid solution. """
    if len(sudoku) != 9:
        return False

    # Check rows
    for row in sudoku:
        if sorted(row) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False

    # Check columns
    for row in range(9):
        temp = []
        for col in range(9):
            temp.append(sudoku[col][row])
        if sorted(temp) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return False

    # Check clusters
    for row in range(3):
        for col in range(3):
            temp = sudoku[row*3][col*3:col*3+3] + sudoku[row*3+1][col*3:col*3+3] + sudoku[row*3+2][col*3:col*3+3]
            if sorted(temp) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                return False

    return True

def same_row(sudoku, row):
    return sudoku[row]

def same_column(sudoku, col):
    return [row[col] for row in sudoku]

def same_cluster(sudoku, row, col):
    return sudoku[row//3*3][col//3*3:col//3*3+3] + sudoku[row//3*3+1][col//3*3:col//3*3+3] + sudoku[row//3*3+2][col//3*3:col//3*3+3]

def solve_sudoku(sudoku, toplevel=True):
    """ Solves the given sudoku with a simple backtrack; nothing fancy. """
    solutions = []
    for row in range(9):
        for col in range(9):
            if sudoku[row][col] == 0:
                for cell in range(1, 10):
                    if not (cell in same_row(sudoku, row) or cell in same_column(sudoku, col) or cell in same_cluster(sudoku, row, col)):
                        temp = copy.deepcopy(sudoku)
                        temp[row][col] = cell
                        solution = solve_sudoku(temp, False)
                        if solution is not None:
                            solutions.append(solution)

                    if toplevel:
                        sys.stdout.write("\rCalculating solution ... "+str(cell*10).rjust(3)+"%")
                        sys.stdout.flush()

                for solution in solutions:
                    if is_valid_solution(solution):
                        return solution
                return None

    if is_valid_solution(sudoku):
        return sudoku
    return None

def main():
    if len(sys.argv) < 2:
        print("Please supply a path to an image file as an argument.")
        sys.exit(1)

    try:
        img = cv2.imread(sys.argv[1], 1)
        assert img is not None
    except:
        print("Could not open image. Please make sure that the file you specified exists and is a valid image file.")
        sys.exit(1)

    # Resize the image to a more appropriate size
    if img.shape[0] > img.shape[1]:
        sizecoef = 800 / img.shape[0]
    else:
        sizecoef = 800 / img.shape[1]
    if sizecoef < 1:
        img = cv2.resize(img, (0, 0), fx=sizecoef, fy=sizecoef)

    debug_image(img, "img")

    sys.stdout.write("Preparing image ...")
    sys.stdout.flush()
    if DEBUG:
        projection = project_image(img)
    else:
        try:
            projection = project_image(img)
        except:
            print("\rPreparing image ... failed.")
            sys.exit(1)
    print("\rPreparing image ... done.")

    sys.stdout.write("Extracting data ...")
    sys.stdout.flush()
    if DEBUG:
        sudoku = extract_sudoku(projection)
    else:
        try:
            sudoku = extract_sudoku(projection)
        except:
            print("\rExtracting data ... failed.")
            sys.exit(1)
    print("\rExtracting data ... done.")

    print("")
    print_sudoku(sudoku)
    print("")

    sys.stdout.write("Calculating solution ...")
    sys.stdout.flush()
    if DEBUG:
        solution = solve_sudoku(sudoku)
    else:
        try:
            solution = solve_sudoku(sudoku)
        except:
            print("\rCalculating solution ... failed.")
            sys.exit(1)
    print("\rCalculating solution ... done.")

    print("")
    print_sudoku(solution)
    print("")


if __name__ == "__main__":
    main()
