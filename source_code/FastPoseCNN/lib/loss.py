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
# Quaternion specific loss function

class PixelWiseQLoss(_Loss):

    def __init__(self, key, eps=0.001):
        super(PixelWiseQLoss, self).__init__()
        self.key = key
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:

        # Selecting the categorical_mask
        cat_mask = pred['auxilary']['cat_mask']

        # Creating union mask between pred and gt
        mask_union = torch.logical_and(cat_mask != 0, gt['mask'] != 0)

        # Return 0.5 if no matching between masks
        if torch.sum(mask_union) == 0:
            return torch.tensor(float('nan'), device=cat_mask.device)

        # Access the predictions to calculate the loss (NxAxHxW) A = [3,4]
        gt_q = gt[self.key]
        pred_q = pred[self.key]

        # Normalizing the masked predicted quaternions
        q = torch.div(pred_q, pred_q)

        # Apply the QLoss Function
        # log(\epsilon + 1 - |gt_q dot pred_q|)
        # gt_q dot pred_q = a1*b1 + a2*b2 + a3*b3 + a4*b4
        """
        dot_product = gt_q[:,0] * q[:,0] + gt_q[:,1] * q[:,1] + gt_q[:,2] * q[:,2] + gt_q[:,3] * q[:,3]
        mag_dot_product = torch.abs(dot_product)
        difference = self.eps + 1 - mag_dot_product
        log_difference = -torch.log(difference)

        return torch.mean(log_difference)
        """
        dot_product = gt_q[:,0] * q[:,0] + gt_q[:,1] * q[:,1] + gt_q[:,2] * q[:,2] + gt_q[:,3] * q[:,3]
        mag_dot_product = torch.pow(dot_product, 2)
        error = self.eps + 1 - mag_dot_product
        loss = torch.log(error + self.eps) - torch.log(torch.tensor(self.eps, device=error.device))

        # Mask the loss based on the union mask
        loss[mask_union == False] = 0
        masked_loss = torch.sum(loss) / torch.sum(mask_union)

        return masked_loss

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

        # Container for per class collective loss
        per_class_loss = []

        # Apply the QLoss Function 
        # log(\epsilon + 1 - |gt_q dot pred_q|)
        for class_id in range(len(gt_pred_matches)):

            # Catching no-instance scenario
            if self.key not in gt_pred_matches[class_id].keys():
                continue

            # Selecting the ground truth data
            gt = gt_pred_matches[class_id][self.key][0]

            # Selecting the predicted data
            pred = gt_pred_matches[class_id][self.key][1]

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

            # Storing the loss per class
            per_class_loss.append(loss)

        # Concatenate the losses to later sum the loss
        # If not empty
        if per_class_loss:
            cat_losses = torch.cat(per_class_loss)
        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches[0]['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        cat_losses = cat_losses[torch.isnan(cat_losses) == False]

        # Return the some of all the losses
        return torch.mean(cat_losses)

#-------------------------------------------------------------------------------
# Pose loss functions

class Iou3dLoss(_Loss):  

    def __init__(self, eps=0.1):
        super(Iou3dLoss, self).__init__()
        self.eps = eps

    def forward(self, gt_pred_matches) -> Tensor:

        # Container for per class collective loss
        per_class_loss = []

        for class_id in range(len(gt_pred_matches)):

            # Catching no-instance scenario
            if 'RT' not in gt_pred_matches[class_id].keys():
                continue

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches[class_id]['RT'][0]
            gt_scales = gt_pred_matches[class_id]['scales'][0]
            pred_RTs = gt_pred_matches[class_id]['RT'][1]
            pred_scales = gt_pred_matches[class_id]['scales'][1]

            # Calculating the iou 3d for between the ground truth and predicted 
            ious_3d = gtf.get_3d_ious(gt_RTs, pred_RTs, gt_scales, pred_scales)

            # Calculating the error
            error = 1 - ious_3d

            # Calculating the loss
            loss = error
            #loss = torch.log(error + self.eps) - torch.log(torch.tensor(self.eps, device=error.device))

            # Storing the loss per class
            per_class_loss.append(loss)

        # Concatenate the losses to later sum the loss
        # If not empty
        if per_class_loss:
            cat_losses = torch.cat(per_class_loss)
        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches[0]['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        cat_losses = cat_losses[torch.isnan(cat_losses) == False]

        # Return the some of all the losses
        return torch.mean(cat_losses)

class OffsetLoss(_Loss):  

    def __init__(self, eps=0.1):
        super(OffsetLoss, self).__init__()
        self.eps = eps

    def forward(self, gt_pred_matches) -> Tensor:

        # Container for per class collective loss
        per_class_loss = []

        for class_id in range(len(gt_pred_matches)):

            # Catching no-instance scenario
            if 'scales' not in gt_pred_matches[class_id].keys():
                continue

            # Grabbing the gt and pred (RT and scales)
            gt_RTs = gt_pred_matches[class_id]['RT'][0]
            pred_RTs = gt_pred_matches[class_id]['RT'][1]

            # Determing the offset errors
            offset_errors = gtf.from_RTs_get_T_offset_errors(
                gt_RTs,
                pred_RTs
            )

            # Calculating the loss
            #loss = torch.log(offset_errors + self.eps) - torch.log(torch.tensor(self.eps, device=offset_errors.device))
            loss = offset_errors / 10

            # Storing the loss per class
            per_class_loss.append(loss)

        # Concatenate the losses to later sum the loss
        # If not empty
        if per_class_loss:
            cat_losses = torch.cat(per_class_loss)
        else:
            try:
                return torch.tensor(float('nan'), device=gt_pred_matches[0]['instance_masks'].device).float()   
            except:
                return torch.tensor(float('nan')).cuda().float()   

        # Remove any nans in the data
        cat_losses = cat_losses[torch.isnan(cat_losses) == False]

        # Return the some of all the losses
        return torch.mean(cat_losses)







