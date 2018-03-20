# CSC320 Winter 2018
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
import random
# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # loop over each pixel
    N, M = source_patches.shape[0], source_patches.shape[1]
    best_D = np.ones((N, M)) * ((255 ** 2) * 3)
    if odd_iteration:
        x_s, x_e, y_s, y_e = 1, N-2, 1, M-2
        step = 1
    else:
        x_s, x_e, y_s, y_e = N-2, 1, M-2, 1
        step = -1
        # odd iteration
        # use f(x-1, y) and f(x, y-1) as offsets
    for i in range(x_s, x_e, step):
        for j in range(y_s, y_e, step):
            this_x, this_y = i + new_f[i, j][0], j + new_f[i, j][1]
            D_this_loop = compute_distance(source_patches[i, j], target_patches[this_x, this_y])
            if D_this_loop < best_D[i, j]:
                best_D[i, j] = D_this_loop
            if not propagation_enabled:
                # init the starting patch location
                starts_i = np.clip(i + new_f[i, j][0], -N, N-1)
                starts_j = np.clip(j + new_f[i, j][1], -M, M-1)
                D_this_loop = compute_distance(source_patches[i, j],
                                               target_patches[starts_i,
                                            starts_j])
                potential_1, potential_2 = new_f[i+step, j], new_f[i, j+step]
                if potential_1[0] + i < N and potential_1[1] + j < M:
                    target_x, target_y = np.clip(i + potential_1[0], -N, N-1),\
                                         np.clip(j + potential_1[1], -M, M-1)
                    D_potential_1 = compute_distance(source_patches[i, j],
                                                target_patches[target_x, target_y])
                    if D_potential_1 < best_D[i, j]:
                        best_D[i, j] = D_potential_1
                        new_f[i, j] = potential_1
                else:
                    D_potential_1 = np.nan
                if potential_2[0] + i < N and potential_2[1] + j < M:
                    target_x, target_y = np.clip(i + potential_2[0], -N, N-1),\
                                         np.clip(j + potential_2[1], -M, M-1)
                    D_potential_2 = compute_distance(source_patches[i, j],
                                                target_patches[target_x, target_y])
                    if D_potential_2 < best_D[i, j]:
                        best_D[i, j] = D_potential_2
                        new_f[i, j] = potential_2
                else:
                    D_potential_2 = np.nan
                # new_Ds = [D_potential_1, D_potential_2, D_this_loop]
                # F_candidate = [potential_1, potential_2, f[i, j]]
                # min_ = np.min(new_Ds)
                # ind = new_Ds.index(min_)
                # if best_D[i, j] == np.nan:
                #     best_D[i, j] = min_
                #     f[i, j] = F_candidate[ind]
                #     new_f[i, j] = F_candidate[ind]
                # if min_ < best_D[i, j]:
                #     best_D[i, j] = min_
                #     f[i, j] = F_candidate[ind]
                #     new_f[i, j] = F_candidate[ind]
            if not random_enabled:
                k = 0
                r = w * (alpha ** k)
                v = new_f[i, j]
                while (r >= 1):
                    # candidate_1 = (np.array([i, j]) + v)[0]
                    # candidate_2 = target_patches.shape[0] - candidate_1 - 1
                    # candidate_3 = (np.array([i, j]) + v)[1]
                    # candidate_4 = target_patches.shape[1] - candidate_3 - 1
                    # new_r = max(r, candidate_1, candidate_2, candidate_3, candidate_4)
                    if r <= 0:
                        break
                    R = np.random.uniform(-1, 1, 1*2) * r + v
                    random_x, random_y = np.int(i + R[0]), np.int(j + R[1])
                    random_x = np.clip(random_x, -N, N - 1)
                    random_y = np.clip(random_y, -M, M - 1)
                    random_distance = compute_distance(source_patches[i, j], target_patches[random_x, random_y])
                    if random_distance < best_D[i, j]:
                        best_D[i, j] = random_distance
                        new_f[i, j] = R
                        f[i, j] = R
                    k += 1
                    r = w * (alpha ** k)
    # else:
    #     # even iteration
    #     for i in range(N-1, 1, -1):
    #         for j in range(M-1, 1, -1):
    #             D_this_loop = None
    #             if not propagation_enabled:
    #                 # init the starting patch location
    #                 starts_i = np.clip(i + new_f[i, j, 0], -N, N-1)
    #                 starts_j = np.clip(j + new_f[i, j, 1], -M, M-1)
    #                 D_this_loop = compute_distance(source_patches[i, j],
    #                                                target_patches[starts_i,
    #                                             starts_j])
    #                 potential_1, potential_2 = new_f[i-1, j], new_f[i, j-1]
    #                 if potential_1[0] + i < N and potential_1[1] + j < M:
    #                     target_x, target_y = np.clip(i + potential_1[0], -N, N-1),\
    #                                          np.clip(j + potential_1[1], -M, M-1)
    #                     D_potential_1 = compute_distance(source_patches[i, j],
    #                                                 target_patches[target_x, target_y])
    #                     if D_potential_1 < best_D[i, j]:
    #                         best_D[i, j] = D_potential_1
    #                         new_f[i, j] = potential_1
    #                 else:
    #                     D_potential_1 = np.nan
    #                 if potential_2[0] + i < N and potential_2[1] + j < M:
    #                     target_x, target_y = np.clip(i + potential_2[0], -N, N-1),\
    #                                          np.clip(j + potential_2[1], -M, M-1)
    #                     D_potential_2 = compute_distance(source_patches[i, j],
    #                                                 target_patches[target_x, target_y])
    #                     # if D_potential_2 < best_D[i, j]:
    #                     #     best_D[i, j] = D_potential_2
    #                     #     new_f[i, j] = potential_2
    #                 else:
    #                     D_potential_2 = np.nan
    #                 new_Ds = [D_potential_1, D_potential_2, D_this_loop]
    #                 F_candidate = [potential_1, potential_2, new_f[i, j]]
    #                 print(new_Ds)
    #                 min_ = np.min(new_Ds)
    #                 ind = new_Ds.index(min_)
    #                 if best_D[i, j] == np.nan:
    #                     best_D[i, j] = min_
    #                     f[i, j] = F_candidate[ind]
    #                     new_f[i, j] = F_candidate[ind]
    #                 if min_ < best_D[i, j]:
    #                     best_D[i, j] = min_
    #                     f[i, j] = F_candidate[ind]
    #                     new_f[i, j] = F_candidate[ind]
    #             # random search
    #             if not random_enabled:
    #                 k = 0
    #                 r = w * (alpha ** k)
    #                 v = new_f[i, j]
    #                 while (r >= 1):
    #                     candidate_1 = (np.array([i, j]) + v)[0]
    #                     candidate_2 = target_patches.shape[0] - candidate_1 - 1
    #                     candidate_3 = (np.array([i, j]) + v)[1]
    #                     candidate_4 = target_patches.shape[1] - candidate_3 - 1
    #                     new_r = max(r, candidate_1, candidate_2, candidate_3, candidate_4)
    #                     if new_r <= 0:
    #                         break
    #                     R = np.random.uniform(-1, 1, 1*2) * new_r + v
    #                     random_x, random_y = np.int(i + R[0]), np.int(j + R[1])
    #                     random_x = np.clip(random_x, -N, N - 1)
    #                     random_y = np.clip(random_y, -M, M - 1)
    #                     random_distance = compute_distance(source_patches[i, j], target_patches[random_x, random_y])
    #                     if random_distance < best_D[i, j]:
    #                         best_D[i, j] = random_distance
    #                         new_f[i, j] = R
    #                         f[i, j] = R
    #                     k += 1
    #                     r = w * (alpha ** k)

    #############################################
    return new_f, best_D, global_vars


# helper function for computing D
def compute_distance(a, b):
    distance = np.abs(a.flatten() - b.flatten())
    return np.sum(distance)

# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    cord_matrix = make_coordinates_matrix(target.shape) + f
    rec_source = target[cord_matrix[:, :, 0], cord_matrix[:, :, 1]]

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
