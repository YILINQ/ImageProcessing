## CSC320 Winter 2018
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions,
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################
from numpy.linalg import inv


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing
    # algorithms. These images are initialized to None and populated/accessed by
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self):
        return {
            'backA': {'msg': 'Image filename for Background A Color', 'default': None},
            'backB': {'msg': 'Image filename for Background B Color', 'default': None},
            'compA': {'msg': 'Image filename for Composite A Color', 'default': None},
            'compB': {'msg': 'Image filename for Composite B Color', 'default': None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut': {'msg': 'Image filename for Object Color', 'default': ['color.tif']},
            'alphaOut': {'msg': 'Image filename for Object Alpha', 'default': ['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn': {'msg': 'Image filename for Object Color', 'default': None},
            'alphaIn': {'msg': 'Image filename for Object Alpha', 'default': None},
            'backIn': {'msg': 'Image filename for Background Color', 'default': None},
        }

    def compositingOutput(self):
        return {
            'compOut': {'msg': 'Image filename for Composite Color', 'default': ['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the
    # matting instance's private dictionary object. The key
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if key not in self._images.keys():
            msg = "Wrong key"
        # elif not cv.imread(fileName):
        #     msg = "Wrong fileName"
        elif key != "alphaIn":
            success = True
            img = cv.imread(fileName)
            img.astype(np.float32)
            self._images[key] = img
            msg = "Read success"

        elif key == "alphaIn":
            img = cv.imread(fileName)
            img.astype(np.float32)
            self._images["alphaIn"] = img
            success = True
            msg = "Read success"
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63.
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if key == "colOut":
            success = True
            msg = "Write success"
            img = self._images["colOut"]
            img.astype(np.float32)
            cv.imwrite(fileName, img)

        if key == "alphaOut":
            success = True
            msg = "Write success"
            img = self._images["alphaOut"]
            img.astype(np.float32)
            cv.imwrite(fileName, img)

        if key == "compOut":
            success = True
            msg = "Write success"
            img = self._images["compOut"]
            cv.imwrite(fileName, img)
        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary
    # ojbect.
    def triangulationMatting(self):
        """
        success, errorMessage = triangulationMatting(self)

        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.

        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if self._images["backA"].shape != self._images["backB"].shape or self._images["compA"].shape != self._images["compB"].shape or self._images["compA"].shape != self._images["compB"].shape or self._images["compA"].shape != self._images["compB"].shape or self._images["compA"].shape != self._images["compB"].shape or self._images["compA"].shape != self._images["compB"].shape:
            success = False
            msg = "Input files have different sizes."

        elif (self._images["backA"] is not None) and (self._images["backB"] is not None) \
                and (self._images["compA"] is not None) and (self._images["compB"] is not None):
            # four rgb arrays
            success = True
            backA = self._images["backA"] / 255.0
            backA.astype(np.float32)
            backB = self._images["backB"] / 255.0
            backB.astype(np.float32)
            compA = self._images["compA"] / 255.0
            compA.astype(np.float32)
            compB = self._images["compB"] / 255.0
            compB.astype(np.float32)

            backA_R = backA[:, :, 2]
            backB_R = backB[:, :, 2]

            compA_R = compA[:, :, 2]
            compA_G = compA[:, :, 1]
            compA_B = compA[:, :, 0]
            compB_R = compB[:, :, 2]
            compB_G = compB[:, :, 1]
            compB_B = compB[:, :, 0]

            backA_G = backA[:, :, 1]
            backB_G = backB[:, :, 1]

            backA_B = backA[:, :, 0]
            backB_B = backB[:, :, 0]


            alpha = 1.0 - (((compA_R - compB_R) * (backA_R - backB_R) +
                            (compA_G - compB_G) * (backA_G - backB_G) +
                            (compA_B - compB_B) * (backA_B - backB_B)) /
                           ((backA_R - backB_R) ** 2 + (backA_G - backB_G) ** 2 + (backA_B - backB_B) ** 2))
            # backA .. comB = image
            # four rgb arrays
            # assume they have the same shape
            # height = backA.shape[0]
            # width = backA.shape[1]

            colOut_R = compA_R - backA_R + alpha * backA_R
            colOut_G = compA_G - backA_G + alpha * backA_G
            colOut_B = compA_B - backA_B + alpha * backA_B
            colOut = cv.merge((colOut_B, colOut_G, colOut_R))
            # for i in range(height):
            #     for j in range(width):
            #         A = np.array([[1, 0, 0, -backA[i, j][2]],
            #                      [0, 1, 0, -backA[i, j][1]],
            #                      [0, 0, 1, -backA[i, j][0]],
            #                      [1, 0, 0, -backB[i, j][2]],
            #                      [0, 1, 0, -backB[i, j][1]],
            #                      [0, 0, 1, -backB[i, j][0]]])
            #         B = np.array([[compA[i, j][2] - backA[i, j][2]],
            #                      [compA[i, j][1] - backA[i, j][1]],
            #                      [compA[i, j][0] - backA[i, j][0]],
            #                      [compB[i, j][2] - backB[i, j][2]],
            #                      [compB[i, j][1] - backB[i, j][1]],
            #                      [compB[i, j][0] - backB[i, j][0]]])
            #         x = np.matmul(np.matmul(inv(np.matmul(A.transpose(), A)), A.transpose()), B)
            #         '''
            #         AT = A.transpose()
            #         ATA = np.matmul(AT, A)
            #         ATA_inv = inv(ATA)
            #         ATA_inv_AT = np.matmul(ATA_inv, AT)
            #         x = np.matmul(ATA_inv_AT, B)
            #         '''
            #         colOut[i, j] = np.array([[x[2][0], x[1][0], x[0][0]]])
            #         alpha[i, j] = x[3]

            alpha = alpha * 255.0
            colOut *= 255.0
            self._images["colOut"] = colOut
            self._images["alphaOut"] = alpha


        else:
            msg = "Missing pictures"
        #########################################

        return success, msg

    def createComposite(self):
        """
success, errorMessage = createComposite(self)

        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if (self._images["backIn"] is not None) and (self._images["colIn"] is not None) and (
            self._images["alphaIn"] is not None):
            self.useTriangulationResults()
            msg = "Composite success"
            back = self._images["backIn"] / 255.0
            col = self._images["colIn"] / 255.0
            alpha = self._images["alphaIn"] / 255.0
            col_R = col[:, :, 2]
            col_G = col[:, :, 1]
            col_B = col[:, :, 0]
            back_R = back[:, :, 2]
            back_G = back[:, :, 1]
            back_B = back[:, :, 0]

            compOut = col + back - alpha * back
            success = True
            # m = self._images["colIn"].shape[0]
            # n = self._images["colIn"].shape[1]
            # img_out = self._images["colIn"]
            # for i in range(m):
            #     for j in range(n):
            #         compOut_R = self._images["colIn"][i][j][2] + \
            #         (1.0 - self._images["alphaIn"][i, j] / 255.0) * self._images["backIn"][i, j][2]
            #         compOut_G = self._images["colIn"][i][j][1] + \
            #         (1.0 - self._images["alphaIn"][i, j] / 255.0) * self._images["backIn"][i, j][1]
            #         compOut_B = self._images["colIn"][i][j][0] + \
            #         (1.0 - self._images["alphaIn"][i, j] / 255.0) * self._images["backIn"][i, j][0]
            #
            #         img_out[i, j] = [compOut_B[0], compOut_G[0], compOut_R[0]]
            self._images["compOut"] = compOut * 255.0
            # self._images["compOut"] = cv.merge((out_B, out_G, out_R))
        #########################################

        return success, msg
