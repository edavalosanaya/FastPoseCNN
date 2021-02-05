import os
import sys
import time
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import scipy.special

# Local imports
sys.path.append(os.getenv("TOOLS_DIR"))

try:
    import visualize as vz
except ImportError:
    pass

import gpu_tensor_funcs as gtf

#-------------------------------------------------------------------------------
# Primary Hough Voting Routines

class HoughVotingLayer(nn.Module):
    
    def __init__(self, HPARAM):
        super().__init__()
        self.HPARAM = HPARAM

    def forward(self, agg_data):

        # Obtain data needed for hough voting
        uv_img = agg_data['xy']
        mask = agg_data['instance_masks']
        
        # Performing hough voting
        output = self.batchwise_hough_voting(uv_img, mask)

        # Store data
        agg_data.update(output)

        return agg_data

    #---------------------------------------------------------------------------
    # Hough Voting per batch

    def batchwise_hough_voting(self, uv_img, mask):

        # If instances exist, perform hough voting
        if uv_img.shape[0] != 0:

            # Generate hypothesis
            hypothesis, all_pts = self.batchwise_generate_hypothesis(
                uv_img, 
                mask
            )

            # Pruning of outliers
            pruned_hypothesis = self.prun_outliers(hypothesis)

            # Calculate the weights of each hypothesis
            weights = self.batchwise_calculate_hypothesis_weights(
                all_pts, 
                uv_img, 
                pruned_hypothesis
            )

            # Account for the pruning of outliers
            is_nan = torch.isnan(pruned_hypothesis)
            is_outlier = torch.squeeze(torch.logical_or(is_nan[:,:,0], is_nan[:,:,1]), dim=-1)
            pruned_hypothesis[is_nan] = 0
            weights[is_outlier] = 0

            # Calculate the weighted means
            weighted_mean = torch.sum(pruned_hypothesis * torch.unsqueeze(weights, dim=-1), dim=1)

            # Need to flip xy to yx
            pixel_xy = weighted_mean[:,[1,0]]

            # Visualize hough voting
            """
            plot = vz.visualize_hypothesis(
                hypothesis,
                pruned_hypothesis,
                pixel_xy,
                uv_img,
                mask
            )
            """

            # Put all valuable data into dictionary
            output = {
                'xy': pixel_xy,
                'xy_mask': uv_img,
                'hypothesis': hypothesis,
                'pruned_hypothesis': pruned_hypothesis
            }

        else:

            # Output for no instances
            output = {
                'xy': torch.zeros((0, 2), device=uv_img.device),
                'xy_mask': uv_img,
                'hypothesis': torch.zeros((0, self.HPARAM.HV_NUM_OF_HYPOTHESES, 2), device=uv_img.device),
                'pruned_hypothesis': torch.zeros((0, self.HPARAM.HV_NUM_OF_HYPOTHESES, 2), device=uv_img.device),
            }

        return output

    def batchwise_generate_hypothesis(self, uv_img, mask):

        valid_samples = []
        single_pt_hypothesis = {}
        all_pts = []

        for i in range(uv_img.shape[0]):

            # Obtain the pts of the mask
            pts = torch.stack(torch.where(mask[i]), dim=1)

            # Determining the number of pts present
            num_of_pts = pts.shape[0]

            # Store the points to avoid recalculating
            all_pts.append(pts)

            # Determine if this is a valid number of pts (>1)
            if num_of_pts < 2:
                single_pt_hypothesis[i] = pts.expand((self.HPARAM.HV_NUM_OF_HYPOTHESES,2))
                continue

            # Selecting random pairs of pts (0-n) [n = number of pts]
            # By using torch.multinomial: https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial

            # first creating uniform probability distribution
            weights = torch.ones(num_of_pts, device=uv_img.device).expand(self.HPARAM.HV_NUM_OF_HYPOTHESES, -1)

            # Obtaining the point pairs via sequence numbers: (N, 2)
            point_pair_idx = torch.multinomial(weights, num_samples=2, replacement=False)

            # Indexing the pts locations among the image (random_pt_pairs: 2xNx2)
            # first index = (pair division), second index = (pt index), third index = (pt's x and y)
            pt_pairs = torch.stack([pts[point_pair_idx[:,0]], pts[point_pair_idx[:,1]]])

            # Indexing the pts unit vector values
            uv_pt_pairs = uv_img[i, :, pt_pairs[:,:,0], pt_pairs[:,:,1]]
            uv_pt_pairs = uv_pt_pairs.permute(1,2,0)

            # Construct the system of equations
            A = torch.stack((uv_pt_pairs[0], -uv_pt_pairs[1]), dim=-1)
            B = torch.stack((-pt_pairs[0], pt_pairs[1]), dim=-1)
            B = (B[:,:,0] + B[:,:,1]).reshape((pt_pairs.shape[1],-1,1))

            if len(valid_samples) == 0:
                total_A = A
                total_B = B
                total_pt_pairs = pt_pairs
                total_uv_pt_pairs = uv_pt_pairs
            else:
                total_A = torch.cat((total_A, A), dim=0)
                total_B = torch.cat((total_B, B), dim=0)
                total_pt_pairs = torch.cat((total_pt_pairs, pt_pairs), dim=1)
                total_uv_pt_pairs =torch.cat((total_uv_pt_pairs, uv_pt_pairs), dim=1)

            # Keep track of the idx of valid entries
            valid_samples.append(i)

        if valid_samples:
            # Perform solver for system of linear system of equations to find
            # intersection of the pts and vectors
            valid_Y = self.batched_pinverse_solver(
                total_A, 
                total_B, 
                total_pt_pairs, 
                total_uv_pt_pairs
            )

            # Split the valid hypothesis and place them in a indentifying dictionary
            chunked_valid_Y = torch.chunk(valid_Y, len(valid_samples))
            valid_samples = {k:v for k,v in zip(valid_samples, chunked_valid_Y)}

        else:
            # Simply to be consistant in data type for latter part of the algorithm.
            valid_samples = {}

        # Splitting the total_Y based on the number of instances
        total_Y = torch.zeros(
            (uv_img.shape[0], self.HPARAM.HV_NUM_OF_HYPOTHESES, 2),
            device=uv_img.device
        )

        # Combine all the hypothesis into a single tensor
        for i in range(uv_img.shape[0]):

            # Check if this is a valid sample
            if i in valid_samples.keys():
                total_Y[i] = valid_samples[i]
            elif i in single_pt_hypothesis.keys():
                total_Y[i] = single_pt_hypothesis[i]
            else:
                raise RuntimeError("Invalid key in hough votingsss")

        return total_Y, all_pts

    def batchwise_calculate_hypothesis_weights(self, all_pts, uv_img, hypothesis):

        all_weights = []

        # Determine the size
        for i in range(uv_img.shape[0]):

            h = hypothesis[i]
            pts = all_pts[i]

            n_of_h = h.shape[0]
            n_of_p = pts.shape[0]

            pts_uv_value = uv_img[i, :, pts[:,0], pts[:,1]]
            pts_uv_value = pts_uv_value.permute(1,0)

            # Expand data to prepare large computation
            expanded_hypo = torch.unsqueeze(h, dim=1).expand((n_of_h, n_of_p, 2))
            expanded_pts = pts.expand((n_of_h, n_of_p, 2))
            pts_uv_value = pts_uv_value.expand((n_of_h, n_of_p, 2))

            # Calculate weight
            a = (expanded_hypo - expanded_pts)
            a = a / torch.unsqueeze(a.norm(dim=-1), dim=-1)
            b = torch.einsum('ijk,ijk->ij', a, pts_uv_value) > 0
            weights = torch.sum(b, dim=-1)

            # Multiply the weight if the hypothesis is inside the mask
            match = (expanded_hypo.long() == expanded_pts)
            xy_match = torch.logical_and(match[...,0], match[...,1])
            h_in_mask = torch.sum(xy_match, dim=1)
            factor = torch.where(
                h_in_mask == 1, 
                self.HPARAM.HV_HYPOTHESIS_IN_MASK_MULTIPLIER,
                1
            )
            weights = factor * weights

            # Normalizing weight
            weights = weights / max(torch.sum(weights), 1)

            # Store weights
            all_weights.append(weights)

        # Stack the results
        all_weights = torch.stack(all_weights)

        return all_weights

    #---------------------------------------------------------------------------
    # Hough Voting per single input

    def hough_voting(self, uv_img, mask):

        # Debugging purposes:
        #h,w = uv_img.shape[-2], uv_img.shape[-1]
        #return torch.tensor([0.5*h, 0.5*w], device=uv_img.device)

        # Determine all the pixel that are in the mask
        pts = torch.stack(torch.where(mask), dim=1)

        # Determining the number of pts present
        num_of_pts = pts.shape[0]

        # If there is less than 2 points, return nan
        if num_of_pts < 2:
            return torch.tensor([float('nan'), float('nan')], device=uv_img.device)

        # Generate hypothesis
        hypothesis = self.generate_hypothesis(uv_img, pts)

        # Check that hypothesese are not just nan
        if torch.isnan(hypothesis).all():
            return torch.tensor([float('nan'), float('nan')], device=uv_img.device)

        # Calculate the weights of each hypothesis
        weights = self.calculate_hypothesis_weights(uv_img, pts, hypothesis)

        # Calculate the weighted means
        weighted_mean = torch.sum(hypothesis * weights.reshape((-1,1)), dim=0) / torch.sum(weights)

        # Need to flip xy to yx
        pixel_xy = torch.tensor(
            [weighted_mean[1], weighted_mean[0]], 
            device=weighted_mean.device, 
            dtype=weighted_mean.dtype
        )

        return pixel_xy

    def generate_hypothesis(self, uv_img, pts):

        # Determining the number of pts present
        num_of_pts = pts.shape[0]

        # Selecting random pairs of pts (0-n) [n = number of pts]
        # By using torch.multinomial: https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial
        
        # limit N based on the maximum number of possible combinations given the 
        # number of points
        max_num_of_pairs = int(scipy.special.comb(num_of_pts, 2))
        N = min(self.HPARAM.HV_NUM_OF_HYPOTHESES, max_num_of_pairs)

        # first creating uniform probability distribution
        weights = torch.ones(num_of_pts, device=uv_img.device).expand(N, -1)

        # Obtaining the point pairs via sequence numbers: (N, 2)
        point_pair_idx = torch.multinomial(weights, num_samples=2, replacement=False)

        # Indexing the pts locations among the image (random_pt_pairs: 2xNx2)
        # first index = (pair division), second index = (pt index), third index = (pt's x and y)
        pt_pairs = torch.stack([pts[point_pair_idx[:,0]], pts[point_pair_idx[:,1]]])

        # Indexing the pts unit vector values
        uv_img = uv_img.permute(1, 2, 0)
        uv_pt_pairs = uv_img[pt_pairs[:,:,0], pt_pairs[:,:,1]]

        # Remove pairs if one of the pts has nan in it.
        is_nan = torch.sum(torch.isnan(uv_pt_pairs), dim=(0,2)) != 0
        
        # If all are nan, then return nan
        if is_nan.all():
            return torch.tensor([float('nan'), float('nan')], device=uv_img.device)
        
        # Elif any nan are present, keep only non_nan values
        elif is_nan.any():
            is_not_nan = ~is_nan
            uv_pt_pairs = uv_pt_pairs[:,is_not_nan,:]
            pt_pairs = pt_pairs[:,is_not_nan,:]

        # Construct the system of equations
        A = torch.stack((uv_pt_pairs[0], -uv_pt_pairs[1]), dim=-1)
        B = torch.stack((-pt_pairs[0], pt_pairs[1]), dim=-1)
        B = (B[:,:,0] + B[:,:,1]).reshape((pt_pairs.shape[1],-1,1))

        # Solving the linear system of equations
        #Y = lstsq_solver(A, B, pt_pairs, uv_pt_pairs)
        Y = self.batched_pinverse_solver(A, B, pt_pairs, uv_pt_pairs)

        # Prun any outliers using std trimming for Y
        Y = self.std_trimming_mean(Y)

        return Y

    def calculate_hypothesis_weights(self, uv_img, pts, hypothesis):

        # Determine the size
        n_of_h = hypothesis.shape[0]
        n_of_p = pts.shape[0]

        uv_img = uv_img.permute(1, 2, 0)
        pts_uv_value = uv_img[pts[:,0], pts[:,1]]

        # Expand data to prepare large computation
        expanded_hypo = torch.unsqueeze(hypothesis, dim=1).expand((n_of_h, n_of_p, 2))
        expanded_pts = pts.expand((n_of_h, n_of_p, 2))
        pts_uv_value = pts_uv_value.expand((n_of_h, n_of_p, 2))

        # Calculate weight
        a = (expanded_hypo - expanded_pts)
        a = a / torch.unsqueeze(a.norm(dim=-1), dim=-1)
        b = torch.einsum('ijk,ijk->ij', a, pts_uv_value) > 0
        weights = torch.sum(b, dim=-1)

        return weights

    #---------------------------------------------------------------------------
    # Linear System of Equations Solvers

    def lstsq_solver(self, A, B, pt_pairs, uv_pt_pairs):

        # Determine the scalar required to make the vectors meet (per pt pair)
        Y = []
        for i in range(pt_pairs.shape[1]):

            # If there is any nan in A, then skip it
            if torch.isnan(A[i]).any():
                continue

            # If issue encountered solving the linear system of equations, skip it
            try:
                xs, qrs = torch.lstsq(A[i].float(), B[i].float())
                
                # Selecting the first x to find the intersection
                x = xs[0] * torch.pow(qrs[0], 2)
            except:
                continue 

            # Using the determine values the intersection point y = mx + b
            b = pt_pairs[0][i]
            m = uv_pt_pairs[0][i]
            y = x[0] * m + b

            # Storing the vectors intersection point
            Y.append(y)

        # Combining the results
        Y = torch.stack(Y)
        return Y

    def batched_pinverse_solver(self, A, B, pt_pairs, uv_pt_pairs):
        """
        Optimized version of lstsq_solver function!
        lstsq_solver: 0.0016908645629882812
        batched_pinverse_solver: 0.0008175373077392578
        """

        # Solving the batch problem
        X = torch.bmm(
            torch.pinverse(A).float(),
            B.float()
        )

        # Selecting the scalar needed for the first pt, to calculate the intersection
        X1 = X[:,0,:]

        # Using the determine values to find the intersection point y = mx + b
        Y = X1 * uv_pt_pairs[0] + pt_pairs[0]

        return Y

    #---------------------------------------------------------------------------
    # Intersection Reduction Functions

    def prun_outliers(self, Y):

        prun_Y = Y.clone()

        # Performed the desired pruning method
        if self.HPARAM.PRUN_METHOD == None:
            return prun_Y
        elif self.HPARAM.PRUN_METHOD == 'z-score':
            outliers_idx = self.batchwise_z_score_trimming(prun_Y)
        elif self.HPARAM.PRUN_METHOD == 'iqr':
            outliers_idx = self.batchwise_iqr_trimming(prun_Y)
        else:
            raise RuntimeError("Invalid HARAM.PRUN_METHOD")

        # Perform the desired behavior on the outliers
        if self.HPARAM.PRUN_OUTLIER_DROP:
            # Fill outliers with Nans
            prun_Y[outliers_idx] = torch.tensor(float('nan'), device=prun_Y.device)
        else:

            # If outliers are to be replaced, then what style (mean, median, mode)
            if self.HPARAM.PRUN_OUTLIER_REPLACEMENT_STYLE == 'mean':
                replace_data = torch.mean(prun_Y, dim=1)
            elif self.HPARAM.PRUN_OUTLIER_REPLACEMENT_STYLE == 'median':
                replace_data = torch.median(prun_Y, dim=1).values

            # Expanding the data to match the size of prun_Y
            expanded_replace_data = torch.unsqueeze(replace_data, dim=1).expand(prun_Y.shape)

            # Filling the outliers with the expanded replacement data
            prun_Y = torch.where(outliers_idx, expanded_replace_data, prun_Y)

        return prun_Y

    def batchwise_z_score_trimming(self, Y):

        # Determing the characteristic of the data
        std = torch.std(Y, dim=1)
        mean = torch.mean(Y, dim=1)

        # Calculate the z score
        diff = (Y - torch.unsqueeze(mean, dim=1))
        z_score = diff / torch.unsqueeze(std, dim=1)

        # Determine the outliers
        outliers = z_score > self.HPARAM.PRUN_ZSCORE_THRESHOLD
        logic_or_outliers = torch.logical_or(outliers[:,:,0], outliers[:,:,1])

        outliers_idx = torch.unsqueeze(logic_or_outliers,dim=-1).expand(Y.shape)

        return outliers_idx

    def batchwise_iqr_trimming(self, Y):

        # Determine Q2
        q2 = torch.median(Y, dim=1).values

        # Creating data container for q1 and q3
        q1 = torch.zeros((Y.shape[0], 2), device=Y.device)
        q3 = torch.zeros((Y.shape[0], 2), device=Y.device)

        # Iterating over the samples
        for i in range(Y.shape[0]):

            # Indexing the specific section we are handling
            y = Y[i]
            i_q2 = q2[i]

            # Find the values below and above
            e_i_q2 = torch.unsqueeze(i_q2, dim=0).expand(y.shape)
            is_lower = y <= e_i_q2
            is_higher = y >= e_i_q2

            # Iterate per axis (x and y)
            for j in range(y.shape[-1]):

                lower_data = y[...,j][is_lower[...,j]]
                higher_data = y[...,j][is_higher[...,j]]

                ij_q1 = torch.median(lower_data)
                ij_q3 = torch.median(higher_data)

                q1[i,j] = ij_q1
                q3[i,j] = ij_q3

        # Calculate IQR score
        iqr = q3 - q1

        # Creating cutoffs (top and bottom)
        top_cut = q3 + self.HPARAM.IQR_MULTIPLIER * iqr
        bot_cut = q1 - self.HPARAM.IQR_MULTIPLIER * iqr

        # Expanding the cutoffs to match the data
        expand_top_cut = torch.unsqueeze(top_cut, dim=1).expand(Y.shape)
        expand_bot_cut = torch.unsqueeze(bot_cut, dim=1).expand(Y.shape)

        # Determine outliers
        high_outliers = Y > expand_top_cut
        low_outliers = Y < expand_bot_cut

        # Determine outliers indecies
        outliers = torch.logical_or(high_outliers, low_outliers)
        logic_or_outliers = torch.logical_or(outliers[:,:,0], outliers[:,:,1])
        outliers_idx = torch.unsqueeze(logic_or_outliers,dim=-1).expand(Y.shape)

        return outliers_idx

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    class DEFAULT_POSE_HPARAM(argparse.Namespace):

        # Algorithmic Parameters
        HV_NUM_OF_HYPOTHESES = 50
        PRUN_GOAL_STD = 25
        PRUN_K_FROM_STD = 1
        PRUN_MAX_ITER = 5

    HPARAM = DEFAULT_POSE_HPARAM()

    # The code below is a simplified version of a real unit vector problem
    hv_layer = HoughVotingLayer(HPARAM)

    # Creating test mask
    mask = torch.ones((5,5))
    mask[0,:] = 0
    mask[-1,:] = 0
    mask[:,0] = 0
    mask[:,-1] = 0

    # Selecting center
    h, w = mask.shape
    center = torch.tensor([2,2])

    # Creating unit vectors
    x_coord = torch.remainder(torch.arange(w*h), w).reshape((h,w)).float()
    y_coord = torch.remainder(torch.arange(w*h), h).reshape((w,h)).float().T
    coord = torch.dstack([y_coord, x_coord])
    diff_norm = torch.norm(center - coord, dim=-1)
    vector = torch.divide((center - coord), torch.unsqueeze(diff_norm, dim=-1))
    vector = vector.permute(2,0,1)

    # Determine the center
    center = hv_layer.forward(vector, mask)
    print(f'Center: {center}')