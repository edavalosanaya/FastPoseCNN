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

            # Calculating the distance between the quaternions
            degree_distance = gtf.torch_quat_distance(q0, q1)

            # Compare against threshold
            thresh_degree_distance = (degree_distance < self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_degree_distance.int())
            self.total = self.total + thresh_degree_distance.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class DegreeAccuracy(pl.metrics.Metric):
    # https://pytorch-lightning.readthedocs.io/en/stable/metrics.html

    def __init__(self):
        super().__init__(f'degree_accuracy')

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

            # Calculating the distance between the quaternions
            degree_distance = gtf.torch_quat_distance(q0, q1)

            # This rounds accuracy
            this_round_accuracy = torch.mean(degree_distance)

            # Update the mean accuracy
            self.accuracy = (self.accuracy + this_round_accuracy) / 2

    def compute(self):
        return self.accuracy

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

        # Obtain all the RT and scales stacked and categorized by class
        stacked_class_RT = gtf.stack_class_matches(true_gt_pred_matches, 'RT')
        stacked_class_scales = gtf.stack_class_matches(true_gt_pred_matches, 'scales')

        # Performing task per class
        for class_number in stacked_class_RT.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = stacked_class_RT[class_number][:,0,:]
            gt_scales = stacked_class_scales[class_number][:,0,:]
            pred_RTs = stacked_class_RT[class_number][:,1,:]
            pred_scales = stacked_class_scales[class_number][:,1,:]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

            # Compare against threshold
            thresh_iou_3d = (ious_3d < self.threshold)

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

        # Obtain all the RT and scales stacked and categorized by class
        stacked_class_RT = gtf.stack_class_matches(true_gt_pred_matches, 'RT')
        stacked_class_scales = gtf.stack_class_matches(true_gt_pred_matches, 'scales')

        # Performing task per class
        for class_number in stacked_class_RT.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = stacked_class_RT[class_number][:,0,:]
            gt_scales = stacked_class_scales[class_number][:,0,:]
            pred_RTs = stacked_class_RT[class_number][:,1,:]
            pred_scales = stacked_class_scales[class_number][:,1,:]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

            # This rounds accuracy
            this_round_accuracy = torch.mean(ious_3d)

            # Update the mean accuracy
            self.accuracy = (self.accuracy + this_round_accuracy) / 2

    def compute(self):
        return self.accuracy

class OffsetAP(pl.metrics.Metric):

    def __init__(self, threshold_cm):
        super().__init__(f'offset_error_mAP_{threshold_cm}cm')
        self.threshold = threshold_cm / 100 # converting from cm to m

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

        # Obtain all the RT stacked and categorized by class
        stacked_class_RT = gtf.stack_class_matches(true_gt_pred_matches, 'RT')

        # Performing task per class
        for class_number in stacked_class_RT.keys():

            # Grabbing the gt and pred RT
            gt_RTs = stacked_class_RT[class_number][:,0,:]
            pred_RTs = stacked_class_RT[class_number][:,1,:]

             # Creating the value for the camera 3d center for later computations
            camera_coord_3d_center = torch.tensor(
                [[0,0,0]], 
                device=gt_RTs.device,
                dtype=gt_RTs.dtype
            ).T

            # Calculating the world centers of the objects (gt and preds)
            # per RT
            gt_world_coord_3d_centers = []
            pred_world_coord_3d_centers = []
            
            for i in range(gt_RTs.shape[0]):

                gt_world_coord_3d_center = gtf.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_center,
                    gt_RTs[i]
                )

                pred_world_coord_3d_center = gtf.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_center,
                    pred_RTs[i]
                )
                
                gt_world_coord_3d_centers.append(gt_world_coord_3d_center)
                pred_world_coord_3d_centers.append(pred_world_coord_3d_center)

            # Combinding all the 3d centers
            gt_world_coord_3d_centers = [x.flatten() for x in gt_world_coord_3d_centers]
            pred_world_coord_3d_centers = [x.flatten() for x in pred_world_coord_3d_centers]

            gt_world_coord_3d_centers = torch.stack(gt_world_coord_3d_centers)
            pred_world_coord_3d_centers = torch.stack(pred_world_coord_3d_centers)

            # Calculating the distance between the gt and pred points
            offset_errors = gtf.get_T_offset_errors(
                gt_world_coord_3d_centers,
                pred_world_coord_3d_centers
            )

            # Compare against threshold
            thresh_offset_error = (offset_errors < self.threshold)

            # Update complete and total
            self.correct = self.correct + torch.sum(thresh_offset_error.int())
            self.total = self.total + thresh_offset_error.shape[0]

    def compute(self):
        return (self.correct.float() / self.total.float()) * 100

class OffsetAccuracy(pl.metrics.Metric):

    def __init__(self):
        super().__init__(f'offset_accuracy')

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

        # Obtain all the RT stacked and categorized by class
        stacked_class_RT = gtf.stack_class_matches(true_gt_pred_matches, 'RT')

        # Performing task per class
        for class_number in stacked_class_RT.keys():

            # Grabbing the gt and pred RT
            gt_RTs = stacked_class_RT[class_number][:,0,:]
            pred_RTs = stacked_class_RT[class_number][:,1,:]

             # Creating the value for the camera 3d center for later computations
            camera_coord_3d_center = torch.tensor(
                [[0,0,0]], 
                device=gt_RTs.device,
                dtype=gt_RTs.dtype
            ).T

            # Calculating the world centers of the objects (gt and preds)
            # per RT
            gt_world_coord_3d_centers = []
            pred_world_coord_3d_centers = []
            
            for i in range(gt_RTs.shape[0]):

                gt_world_coord_3d_center = gtf.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_center,
                    gt_RTs[i]
                )

                pred_world_coord_3d_center = gtf.transform_3d_camera_coords_to_3d_world_coords(
                    camera_coord_3d_center,
                    pred_RTs[i]
                )
                
                gt_world_coord_3d_centers.append(gt_world_coord_3d_center)
                pred_world_coord_3d_centers.append(pred_world_coord_3d_center)

            # Combinding all the 3d centers
            gt_world_coord_3d_centers = [x.flatten() for x in gt_world_coord_3d_centers]
            pred_world_coord_3d_centers = [x.flatten() for x in pred_world_coord_3d_centers]

            gt_world_coord_3d_centers = torch.stack(gt_world_coord_3d_centers)
            pred_world_coord_3d_centers = torch.stack(pred_world_coord_3d_centers)

            # Calculating the distance between the gt and pred points
            offset_errors = gtf.get_T_offset_errors(
                gt_world_coord_3d_centers,
                pred_world_coord_3d_centers
            )

            # Convert the offset (m) to (cm)
            offset_errors *= 100

            # This rounds accuracy
            this_round_accuracy = torch.mean(offset_errors)

            # Update the mean accuracy
            self.accuracy = (self.accuracy + this_round_accuracy) / 2

    def compute(self):
        return self.accuracy 