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
# Pose Losses

class MaskedMSELoss(_Loss):

    def __init__(self, key):
        super(MaskedMSELoss, self).__init__()
        self.key = key

    def forward(self, pred, gt) -> Tensor:

        # Selecting the categorical_mask
        cat_mask = pred['auxilary']['cat_mask']

        # Return 1 if no matching between masks
        if torch.sum(torch.logical_and(cat_mask, pred['gt'])) == 0:
            return torch.tensor([1], device=cat_mask.device, requires_grad=True)

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
            return torch.tensor([0.5], device=cat_mask.device, requires_grad=True).float()

        # Access the predictions to calculate the loss (NxAxHxW) A = [3,4]
        gt_q = gt[self.key]
        pred_q = pred[self.key]

        # Obtaining the magnitude of the quaternion to later normalize
        norm_q = pred_q.norm(dim=1)
        
        # Avoid dividing by zero
        norm_q[norm_q == 0] = 1

        # Expanding the shape of norm_q to match with m_pred_q
        norm_q = torch.unsqueeze(norm_q, dim=1)

        # Normalizing the masked predicted quaternions
        q = torch.div(pred_q, norm_q)

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
        difference = self.eps + 1 - mag_dot_product

        # Mask the difference based on the union mask
        difference[mask_union == False] = 0
        masked_loss = torch.sum(difference) / torch.sum(mask_union)

        return masked_loss

class AggregatedQLoss(_Loss):
    """AggregatedQLoss

    References:
    https://math.stackexchange.com/questions/90081/quaternion-distance
    http://kieranwynn.github.io/pyquaternion/#normalisation
    https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L800
    


    Args:
        _Loss ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, key, eps=0.001):
        super(AggregatedQLoss, self).__init__()
        self.key = key
        self.eps = eps

    def forward(self, matches) -> Tensor:
        """AggregatedQLoss Foward

        Args:
            matches [list]: 
                match ([dict]):
                    class_id: torch.Tensor
                    quaternion: torch.Tensor

        Returns:
            Tensor: [description]
        """

        # Create matches that are only true
        true_gt_pred_matches = []

        # Remove false matches (meaning 'iou_2d_mask')
        for match in matches:

            # Keeping the match if the iou_2d_mask > 0
            if match['iou_2d_mask'] > 0:
                true_gt_pred_matches.append(match)

        # If there is no true matches, simply end update function
        if true_gt_pred_matches == []:
            return torch.tensor([1], requires_grad=True).float().cuda()

        # Stack the matches based on class for all quaternion
        stacked_class_quaternion = gtf.stack_class_matches(true_gt_pred_matches, 'quaternion')

        # Container for per class collective loss
        per_class_loss = {}

        # Apply the QLoss Function 
        # log(\epsilon + 1 - |gt_q dot pred_q|)
        for class_number in stacked_class_quaternion.keys():

            # Selecting the ground truth quaternion
            gt_q = stacked_class_quaternion[class_number][:,0,:]

            # Selecting the predicted quaternion
            pred_q = stacked_class_quaternion[class_number][:,1,:]

            # Normalize the predicted quaternion
            norm_q = torch.div(pred_q.T, pred_q.norm(dim=1).T).T

            # Calculating the loss
            """
            dot_product = torch.diag(torch.mm(gt_q, norm_q.T))
            mag_dot_product = torch.abs(dot_product)
            difference = self.eps + 1 - mag_dot_product
            log_difference = -torch.log(difference)
            per_class_loss[class_number] = log_difference
            """
            dot_product = torch.diag(torch.mm(gt_q, norm_q.T))
            mag_dot_product = torch.pow(dot_product, 2)
            difference = self.eps + 1 - mag_dot_product
            per_class_loss[class_number] = difference

        # Place all the class losses into a single list
        losses = [v for v in per_class_loss.values()]

        # Stack the losses to later sum the loss
        # If not empty
        if losses:
            stacked_losses = torch.cat(losses)
        else:
            for key in stacked_class_quaternion.keys():
                pred_q = stacked_class_quaternion[key]
                try:
                    return -torch.log(torch.tensor([0.5], device=pred_q.device, requires_grad=True))
                except UnboundLocalError:
                    continue
            return torch.tensor([0.5], requires_grad=True).cuda()

        # Return the some of all the losses
        return torch.mean(stacked_losses)
        

            





