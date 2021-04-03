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

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'quaternion' in gt_pred_matches.keys():

            # Determing the degree per error (absolute distance)
            # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L772
            q0 = gt_pred_matches['quaternion'][0]
            q1 = gt_pred_matches['quaternion'][1]

            # Calculating the distance between the quaternions
            degree_distance = gtf.get_quat_distance(q0, q1, gt_pred_matches['symmetric_ids'])

            # Compare against threshold
            thresh_degree_distance = (degree_distance < self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_degree_distance.int())
            self.total = self.total + thresh_degree_distance.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class DegreeError(pl.metrics.Metric):
    # https://pytorch-lightning.readthedocs.io/en/stable/metrics.html

    def __init__(self):
        super().__init__(f'degree_error')

        # Adding state data
        self.add_state('error', default=torch.tensor(0), dist_reduce_fx='mean')

    def update(self, gt_pred_matches):
        """

        Args:
            pred_gt_matches [list]: 
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'quaternion' in gt_pred_matches.keys():

            # Determing the degree per error (absolute distance)
            # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L772
            q0 = gt_pred_matches['quaternion'][0]
            q1 = gt_pred_matches['quaternion'][1]

            # Calculating the distance between the quaternions
            degree_distance = gtf.get_quat_distance(q0, q1, gt_pred_matches['symmetric_ids'])

            # This rounds accuracy
            this_round_error = torch.mean(degree_distance)

            # Update the mean accuracy
            self.error = (self.error + this_round_error) / 2

    def compute(self):
        return self.error

class Iou3dAP(pl.metrics.Metric):

    def __init__(self, threshold):
        super().__init__(f'3D_iou_mAP_{threshold}')
        self.threshold = threshold

        # Adding state data
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, gt_pred_matches):
        """
        Args:
            pred_gt_matches: [list]:
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
                xy: torch.Tensor
                z: torch.Tensor
                scales: torch.Tensor
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches['RT'][0]
            gt_scales = gt_pred_matches['scales'][0]
            pred_RTs = gt_pred_matches['RT'][1]
            pred_scales = gt_pred_matches['scales'][1]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

            # Compare against threshold
            thresh_iou_3d = (ious_3d > self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_iou_3d.int())
            self.total = self.total + thresh_iou_3d.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class Iou3dAccuracy(pl.metrics.Metric):

    def __init__(self):
        super().__init__(f'3D_iou_accuracy')

        # Adding state data
        self.add_state('accuracy', default=torch.tensor(0), dist_reduce_fx='mean')

    def update(self, gt_pred_matches):
        """
        Args:
            pred_gt_matches: [list]:
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
                xy: torch.Tensor
                z: torch.Tensor
                scales: torch.Tensor
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches['RT'][0]
            gt_scales = gt_pred_matches['scales'][0]
            pred_RTs = gt_pred_matches['RT'][1]
            pred_scales = gt_pred_matches['scales'][1]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales) * 100

            # This rounds accuracy
            this_round_accuracy = torch.mean(ious_3d)

            # Update the mean accuracy
            self.accuracy = (self.accuracy + this_round_accuracy) / 2

    def compute(self):
        return self.accuracy

class OffsetAP(pl.metrics.Metric):

    def __init__(self, threshold):
        super().__init__(f'offset_error_mAP_{threshold}cm')
        self.threshold = threshold

        # Adding state data
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, gt_pred_matches):
        """
        Args:
            pred_gt_matches: [list]:
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
                xy: torch.Tensor
                z: torch.Tensor
                scales: torch.Tensor
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred RT
            gt_Ts = gt_pred_matches['T'][0]
            pred_Ts = gt_pred_matches['T'][1]

            # Determing the offset errors
            offset_errors = gtf.from_Ts_get_offset_error(
                gt_Ts,
                pred_Ts
            )

            # Compare against threshold
            thresh_offset_error = (offset_errors < self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_offset_error.int())
            self.total = self.total + thresh_offset_error.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class OffsetError(pl.metrics.Metric):

    def __init__(self):
        super().__init__(f'offset_error')

        # Adding state data
        self.add_state('error', default=torch.tensor(0), dist_reduce_fx='mean')

    def update(self, gt_pred_matches):
        """
        Args:
            pred_gt_matches: [list]:
            match ([dict]):
                class_id: torch.Tensor
                quaternion: torch.Tensor
                xy: torch.Tensor
                z: torch.Tensor
                scales: torch.Tensor
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred RT
            gt_RTs = gt_pred_matches['RT'][0]
            pred_RTs = gt_pred_matches['RT'][1]

            # Determing the offset errors
            offset_errors = gtf.from_RTs_get_T_offset_errors(
                gt_RTs,
                pred_RTs
            )

            # This rounds accuracy
            this_round_error = torch.mean(offset_errors)

            # Update the mean accuracy
            self.error = (self.error + this_round_error) / 2

    def compute(self):
        return self.error 