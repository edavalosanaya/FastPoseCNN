import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.focal import FocalLoss
import gpu_tensor_funcs as gtf

#-------------------------------------------------------------------------------
# Mask Losses

class CE(_Loss):
    """Implementation of CE for 2D model from logits."""

    def __init__(self, ignore_index=-1):
        super(CE, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, gt) -> Tensor:

        # Indexing the corresponding mask pred and ground truth
        y_pred = pred['mask']
        y_true = gt['mask']

        loss = nn.CrossEntropyLoss()

        return loss(y_pred, y_true)

class CCE(_Loss):
    """Implementation of CCE for 2D model from logits."""

    def __init__(self, from_logits=True, ignore_index=-1):
        super(CCE, self).__init__()
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, pred, gt) -> Tensor:
        """
        Args:
            y_pred: NxCxHxW
            y_true: NxHxW
        Returns:
        """

        # Indexing the corresponding mask pred and ground truth
        y_pred = pred['mask']
        y_true = gt['mask']

        y_pred = nn.LogSoftmax(dim=1)(y_pred)

        loss = nn.NLLLoss(ignore_index=self.ignore_index)

        return loss(y_pred, y_true)

class Focal(_Loss):
    """Implementation of Focal for 2D model from logits."""

    def __init__(self, key='mask', from_logits=True, alpha=0.5, gamma=2, ignore_index=-1):
        super(Focal, self).__init__()
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt) -> Tensor:
        """
        Args:
            y_pred: NxCxHxW
            y_true: NxHxW
        Returns:
        """

        # Indexing the corresponding mask pred and ground truth
        y_pred = pred['mask']
        y_true = gt['mask']

        # Converting the data to a more stable range [-inf,0)
        y_pred = nn.LogSoftmax(dim=1)(y_pred)

        # Creating FocalLoss object
        loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, ignore_index=self.ignore_index)

        # Returning the applied FocalLoss object
        return loss(y_pred, y_true)

#-------------------------------------------------------------------------------
# General pixel-wise loss function

class MaskedMSELoss(_Loss):

    def __init__(self, key):
        super(MaskedMSELoss, self).__init__()
        self.key = key

    def forward(self, pred, gt) -> Tensor:

        # Selecting the categorical_mask
        cat_mask = pred['auxilary']['cat_mask']

        # Creating union mask between pred and gt
        mask_union = torch.logical_and(cat_mask != 0, gt['mask'] != 0)

        # Return 1 if no matching between masks
        if torch.sum(mask_union) == 0:
            return torch.tensor(float('nan'), device=cat_mask.device).float()

        # Access the predictions to calculate the loss (NxAxHxW) A = [3,4]
        y_pred = pred[self.key]
        y_gt = gt[self.key]

        # Creating loss function object
        loss_fn = nn.MSELoss()

        # Create overall mask for bg and objects
        binary_pred_mask = (cat_mask != 0)

        # Expand the class_mask if needed
        if len(y_pred.shape) > len(binary_pred_mask.shape):
            binary_pred_mask = torch.unsqueeze(binary_pred_mask, dim=1)

        # Obtain the class specific values from key predictions (quat,scales,ect.)
        masked_pred = y_pred * binary_pred_mask

        """
        # Masking out gradients for the quaternions (from the intersection of the pred and gt masks)
        # https://discuss.pytorch.org/t/masking-out-gradient-before-backpropagating/84418
        boolean_intersection_mask = torch.unsqueeze(torch.logical_and(binary_pred_mask, binary_gt_mask), dim=1)
        
        if masked_pred.requires_grad:
            masked_pred.register_hook(lambda grad: grad * boolean_intersection_mask.float())
        """

        # Calculate the loss
        masked_loss = loss_fn(masked_pred, y_gt)

        return masked_loss

#-------------------------------------------------------------------------------
# General aggregated loss function

class AggregatedLoss(_Loss):
    """AggregatedLoss
    Generic loss function for aggregated data

    Quaternion References:
    https://math.stackexchange.com/questions/90081/quaternion-distance
    http://kieranwynn.github.io/pyquaternion/#normalisation
    https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L800

    Args:
        _Loss ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, key, eps=0.1):
        super(AggregatedLoss, self).__init__()
        self.key = key
        self.eps = eps

    def forward(self, gt_pred_matches) -> Tensor:
        """AggregatedQLoss Foward

        Args:
            matches [list]: 
                match ([dict]):
                    class_id: torch.Tensor

        Returns:
            Tensor: [description]
        """

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss
            if self.key == 'quaternion':
                dot_product = torch.diag(torch.mm(gt, pred.T))
                mag_dot_product = torch.pow(dot_product, 2)
                error = 1 - mag_dot_product
                loss = torch.log(error + self.eps) - torch.log(torch.tensor(self.eps, device=error.device))
            
            elif self.key == 'xy':
                loss = (gt-pred).norm(dim=1) / 10
            
            elif self.key == 'z':
                loss = (torch.log(gt)-torch.log(pred)).norm(dim=1)
            
            elif self.key in ['scales', 'T']:
                loss = (gt-pred).norm(dim=1)
            
            elif self.key == 'R':
                similarity = torch.bmm(torch.transpose(gt, 1, 2), pred)
                traced_value = torch.einsum('bii->b', similarity) # batched trace
                loss = torch.acos((traced_value - 1) / 2)
            
            elif self.key == 'RT':
                loss = (torch.inverse(gt) * pred).norm(dim=1)

            else:
                raise NotImplementedError("Invalid entered key.")
        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)

#-------------------------------------------------------------------------------
# Aggregated loss functions

# Rotation
class QLoss(_Loss): # Quaternion

    def __init__(self, key = None, eps = 0.1):
        super(QLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'quaternion'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Handle symmetric and non-symmetric items differently
            non_symmetric_instances = torch.where(gt_pred_matches['symmetric_ids'] == 0)[0]
            symmetric_instances = torch.where(gt_pred_matches['symmetric_ids'] != 0)[0]

            # Calculate the loss for non-symmetric items
            non_symmetric_loss = self.get_loss(gt[non_symmetric_instances], pred[non_symmetric_instances])

            # Calculate the loss for symmetric items
            symmetric_loss = self.get_symmetric_loss(gt[symmetric_instances], pred[symmetric_instances])

            # Sum the total loss
            loss = torch.cat((non_symmetric_loss, symmetric_loss), dim=0)

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)

    def get_loss(self, gt, pred) -> Tensor:

        # Calculating the loss
        dot_product = torch.diag(torch.mm(gt, pred.T))
        loss = self.dot_product_to_loss(dot_product)

        return loss

    def get_symmetric_loss(self, gt, pred) -> Tensor:

        # HELPFUL RESOURCES ON HOW TO PERFORM THIS OPERATION
        # Batch dot product
        # https://pytorch.org/docs/stable/generated/torch.einsum.html

        # How to perform quaternion multiplication and calculator
        # https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/arithmetic/index.htm#:~:text=The%20multiplication%20rules%20for%20the,%3D%20k*k%20%3D%20-1

        # Visualize quaternions
        # https://quaternions.online

        # Existing implementation of quaternion multiplication!
        # https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.quaternion_multiply

        # Checking if the gt or pred are empty
        if gt.shape[0] == 0:
            return torch.tensor([float('nan')], device=gt.device)

        # ! For DEBUGGING/TESTING 
        #pred = torch.tensor([[0.7071, 0, 0, 0.7071]], device=pred.device)
        #gt = torch.tensor([[0,0,0,1]], device=gt.device)

        # Expanding the pred to account for 0-360 degrees of rotation to account 
        # for z-axis symmetric and expanding the gt to match its size for later 
        # comparison
        rot_e_pred, e_gt = gtf.quat_symmetric_tf(pred, gt)

        # Performing dot product on the transformed e_pred and e_gt
        dot_product = torch.einsum('bij,bij->bi', rot_e_pred.double(), e_gt.double())

        # Determine the maximum along the first dimension (rotated dimension)
        best_alignment = torch.max(dot_product, dim=1).values

        # Calculating the loss
        loss = self.dot_product_to_loss(best_alignment)

        return loss

    def dot_product_to_loss(self, dot_product):

        mag_dot_product = torch.pow(dot_product, 2)
        error = 1 - mag_dot_product
        loss = torch.log(error + self.eps) - torch.log(torch.tensor(self.eps, device=error.device))

        return loss

class RLoss(_Loss): # Rotation Matrix
    
    def __init__(self, key = None, eps = 0.1):
        super(RLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'R'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss
            similarity = torch.bmm(torch.transpose(gt, 1, 2), pred)
            traced_value = torch.einsum('bii->b', similarity) # batched trace
            loss = torch.acos((traced_value - 1) / 2)

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)

# Translation
class TLoss(_Loss): # Translation Vector

    def __init__(self, key = None, eps = 0.1):
        super(TLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'T'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss
            loss = (gt-pred).norm(dim=1)

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)

class XYLoss(_Loss): # 2D Center

    def __init__(self, key = None, eps = 0.1):
        super(XYLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'xy'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss
            abs_diff = torch.abs(gt-pred)

            # Remove any nans in the data
            abs_diff = abs_diff[torch.isnan(abs_diff) == False]

            # Calculating the loss
            loss = torch.mean(abs_diff) / 10 # Reducing the loss by a factor 

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Return the some of all the losses
        return loss

class ZLoss(_Loss): # Z - depth
    
    def __init__(self, key = None, eps = 0.1):
        super(ZLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'z'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss (logging the data first)
            abs_diff = torch.abs(torch.log(gt)-torch.log(pred))

            # Remove any nans in the data
            abs_diff = abs_diff[torch.isnan(abs_diff) == False]

            # Calculating the loss
            loss = torch.mean(abs_diff) # Reducing the loss by a factor

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        return loss

# Scales
class ScalesLoss(_Loss): # h, w, l scales

    def __init__(self, key = None, eps = 0.1):
        super(ScalesLoss, self).__init__()
        self.eps = eps

        if key:
            self.key = key
        else:
            self.key = 'scales'

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and self.key in gt_pred_matches.keys():

            # Selecting the ground truth data
            gt = gt_pred_matches[self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[self.key][1]

            # Calculating the loss
            abs_diff = torch.abs(gt-pred)

            # Remove any nans in the data
            abs_diff = abs_diff[torch.isnan(abs_diff) == False]

            # Calculating the loss
            loss = torch.mean(abs_diff) # Reducing the loss by a factor

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        return loss

#-------------------------------------------------------------------------------
# Pose loss functions

class Iou3dLoss(_Loss):  

    def __init__(self, eps=0.1):
        super(Iou3dLoss, self).__init__()
        self.eps = eps

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches['RT'][0]
            gt_scales = gt_pred_matches['scales'][0]
            pred_RTs = gt_pred_matches['RT'][1]
            pred_scales = gt_pred_matches['scales'][1]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

            # Calculating the error
            error = 1 - ious_3d

            # Calculating the loss
            loss = error
            #loss = torch.log(error + self.eps) - torch.log(torch.tensor(self.eps, device=error.device))

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches[0]['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)

class OffsetLoss(_Loss):  

    def __init__(self, eps=0.1):
        super(OffsetLoss, self).__init__()
        self.eps = eps

    def forward(self, gt_pred_matches) -> Tensor:

        # Catching no-instance scenario
        if type(gt_pred_matches) != type(None) and 'RT' in gt_pred_matches.keys():

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches['RT'][0]
            pred_RTs = gt_pred_matches['RT'][1]

            # Determing the offset errors
            offset_errors = gtf.from_RTs_get_T_offset_errors(
                gt_RTs,
                pred_RTs
            )

            # Calculating the loss
            #loss = torch.log(offset_errors + self.eps) - torch.log(torch.tensor(self.eps, device=offset_errors.device))
            loss = offset_errors / 10

        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches[0]['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        clean_loss = loss[torch.isnan(loss) == False]

        # Return the some of all the losses
        return torch.mean(clean_loss)







