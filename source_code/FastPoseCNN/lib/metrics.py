import torch

import pytorch_lightning as pl

# Local imports
import gpu_tensor_funcs as gtf

#-------------------------------------------------------------------------------
# Classes

class DegreeErrorMeanAP(pl.metrics.Metric):
    # https://pytorch-lightning.readthedocs.io/en/stable/metrics.html

    def __init__(self, threshold):
        super().__init__(f'degree_error_mAP_{threshold}')
        self.threshold = threshold

        # Adding state data
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, gt_pred_matches):
        """

        Args:
            pred_gt_matches [list]: 
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
        """
 
        # Create matches that are only true
        true_gt_pred_matches = []

        # Remove false matches (meaning 'iou_2d_mask')
        for match in gt_pred_matches:

            # Keeping the match if the iou_2d_mask > 0
            if match['iou_2d_mask'] > 0:
                true_gt_pred_matches.append(match)

        # If there is no true matches, simply end update function
        if true_gt_pred_matches == []:
            return

        # obtain all the quaternions stacked and categorized by class
        stacked_class_quaternion = gtf.stack_class_matches(true_gt_pred_matches, 'quaternion')
        
        # Performing task per class
        for class_number in stacked_class_quaternion.keys():

            # Determing the degree per error (absolute distance)
            # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L772
            q0 = stacked_class_quaternion[class_number][:,0,:]
            q1 = stacked_class_quaternion[class_number][:,1,:]

            # Determine the difference
            q0_minus_q1 = q0 - q1
            q0_plus_q1  = q0 + q1
            
            # Obtain the norm
            d_minus = q0_minus_q1.norm(dim=1)
            d_plus  = q0_plus_q1.norm(dim=1)

            # Compare the norms and select the one with the smallest norm
            ds = torch.stack((d_minus, d_plus))
            rad_distance = torch.min(ds, dim=0).values

            # Converting the rad to degree
            degree_distance = torch.rad2deg(rad_distance)

            # Compare against threshold
            thresh_degree_distance = (degree_distance < self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_degree_distance.int())
            self.total = self.total + thresh_degree_distance.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class RotationAccuracy(pl.metrics.Metric):
    # https://pytorch-lightning.readthedocs.io/en/stable/metrics.html

    def __init__(self):
        super().__init__(f'rotation_accuracy')

        # Adding state data
        self.add_state('accuracy', default=torch.tensor(0), dist_reduce_fx='mean')

    def update(self, gt_pred_matches):
        """

        Args:
            pred_gt_matches [list]: 
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
        """

        # Create matches that are only true
        true_gt_pred_matches = []

        # Remove false matches (meaning 'iou_2d_mask')
        for match in gt_pred_matches:

            # Keeping the match if the iou_2d_mask > 0
            if match['iou_2d_mask'] > 0:
                true_gt_pred_matches.append(match)

        # If there is no true matches, simply end update function
        if true_gt_pred_matches == []:
            return

        # obtain all the quaternions stacked and categorized by class
        stacked_class_quaternion = gtf.stack_class_matches(true_gt_pred_matches, 'quaternion')
        
        # Performing task per class
        for class_number in stacked_class_quaternion.keys():

            # Determing the degree per error (absolute distance)
            # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L772
            q0 = stacked_class_quaternion[class_number][:,0,:]
            q1 = stacked_class_quaternion[class_number][:,1,:]

            # Determine the difference
            q0_minus_q1 = q0 - q1
            q0_plus_q1  = q0 + q1
            
            # Obtain the norm
            d_minus = q0_minus_q1.norm(dim=1)
            d_plus  = q0_plus_q1.norm(dim=1)

            # Compare the norms and select the one with the smallest norm
            ds = torch.stack((d_minus, d_plus))
            rad_distance = torch.min(ds, dim=0).values

            # Converting the rad to degree
            degree_distance = torch.rad2deg(rad_distance)

            # This rounds accuracy
            this_round_accuracy = torch.mean(degree_distance)

            # Update the mean accuracy
            self.accuracy = (self.accuracy + this_round_accuracy) / 2

    def compute(self):
        return self.accuracy