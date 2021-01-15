import time

import torch

import scipy.special

import gpu_tensor_funcs as gtf

def hough_voting(uv_img, mask, N=50):

    # Determine all the pixel that are in the mask
    pts = torch.stack(torch.where(mask), dim=1)

    # Determining the number of pts present
    num_of_pts = pts.shape[0]

    # If there is less than 2 points, return nan
    if num_of_pts < 2:
        return torch.tensor([float('nan'), float('nan')], device=uv_img.device).reshape((-1,1))

    # Selecting random pairs of pts (0-n) [n = number of pts]
    # By using torch.multinomial: https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial
    
    # limit N based on the maximum number of possible combinations given the 
    # number of points
    max_num_of_pairs = int(scipy.special.comb(num_of_pts, 2))
    N = min(N, max_num_of_pairs)

    # first creating uniform probability distribution
    weights = torch.ones(num_of_pts, device=uv_img.device).expand(N, -1)

    # While true to ensure that at least one valid point pair is selected
    while True:

        # Obtaining the point pairs via sequence numbers: (N, 2)
        point_pair_idx = torch.multinomial(weights, num_samples=2, replacement=True)

        # Detect pairs with the same pt and remove them
        repeat_pt_pairs = point_pair_idx[:,0] == point_pair_idx[:,1]
        point_pair_idx = point_pair_idx[repeat_pt_pairs == False]

        # If at least one valid point pair is selected, stop searching pairs
        if point_pair_idx.shape[0] != 0:
            break

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
        return torch.tensor([float('nan'), float('nan')], device=uv_img.device).reshape((-1,1))
    
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
    Y = batched_pinverse_solver(A, B, pt_pairs, uv_pt_pairs)

    # Determine the average and standard deviation
    pixel_xy = torch.mean(Y, dim=0).reshape((-1,1))
    std_xy = torch.std(Y, dim=0)

    return pixel_xy

def lstsq_solver(A, B, pt_pairs, uv_pt_pairs):

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

def batched_pinverse_solver(A, B, pt_pairs, uv_pt_pairs):
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

if __name__ == '__main__':

    # The code below is a simplified version of a real unit vector problem

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
    center = hough_voting(vector, mask)
    print(f'Center: {center}')