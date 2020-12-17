import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.focal import FocalLoss
import gpu_tensor_funcs as gtf

#-------------------------------------------------------------------------------
# Mask Losses

class CE(_Loss):
    """Implementation of CE for 2D model from logits."""

    data = 'pixel-wise'

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

    data = 'pixel-wise'

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

    data = 'pixel-wise'

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

    data = 'pixel-wise'

    def __init__(self, key):
        super(MaskedMSELoss, self).__init__()
        self.key = key

    def forward(self, pred, gt) -> Tensor:

        # Obtain ground truth mask (NxHxW)
        gt_mask = gt['mask']
        pred_mask = pred['mask']

        # Access the predictions to calculate the loss (NxAxHxW) A = [3,4]
        y_pred = pred[self.key]
        y_gt = gt[self.key]

        # Creating loss function object
        loss_fn = nn.MSELoss()

        # Create overall mask for bg and objects
        binary_gt_mask = (gt_mask != 0)
        binary_pred_mask = (pred_mask != 0)

        # Expand the class_mask if needed
        if len(y_pred.shape) > len(binary_gt_mask.shape):
            binary_gt_mask = torch.unsqueeze(binary_gt_mask, dim=1)

        # Obtain the class specific values from key predictions (quat,scales,ect.)
        masked_pred = y_pred * binary_gt_mask

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

class QLoss(_Loss):
    """QLoss

    References:
    https://math.stackexchange.com/questions/90081/quaternion-distance
    http://kieranwynn.github.io/pyquaternion/#normalisation
    https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L800
    


    Args:
        _Loss ([type]): [description]

    Returns:
        [type]: [description]
    """

    data = 'matched'

    def __init__(self, key, eps=0.001):
        super(QLoss, self).__init__()
        self.key = key
        self.eps = eps

    def forward(self, matches) -> Tensor:
        """QLoss Foward

        Args:
            matches [list]: 
                match ([dict]):
                    class_id: torch.Tensor
                    quaternion: torch.Tensor

        Returns:
            Tensor: [description]
        """

        # If there is no matches [], skip it
        if not matches:
            return torch.tensor([float('nan')])

        # Given the key, aggregated all the matches of that type by class
        class_quaternion = {}

        # Iterating through all the matches
        for match in matches:

            # If this match is the first of its class object, then add new item
            if int(match['class_id']) not in class_quaternion.keys():
                class_quaternion[int(match['class_id'])] = [match['quaternion']]

            # else, just append it to the pre-existing list
            else:
                class_quaternion[int(match['class_id'])].append(match['quaternion'])

        # Container for per class ground truths and predictions
        stacked_class_quaternion = {}

        # Once the quaternions have been separated by class, stack them all to
        # formally compute the QLoss
        for class_number, class_data in class_quaternion.items():

            # Stacking all matches in one class
            stacked_class_quaternion[class_number] = torch.stack(class_data)

        # Container for per class collective loss
        per_class_loss = {}

        # Apply the QLoss Function 
        # log(\epsilon + 1 - |qbar dot q|)
        for class_number in stacked_class_quaternion.keys():

            # Selecting the ground truth quaternion
            qbar = stacked_class_quaternion[class_number][:,0,:]

            # Selecting the predicted quaternion
            q = stacked_class_quaternion[class_number][:,1,:]

            # Normalize the predicted quaternion
            norm_q = torch.div(q.T, q.norm(dim=1).T).T

            # Calculating the loss
            A = torch.diag(torch.mm(qbar, norm_q.T))
            B = torch.abs(A)
            C = self.eps + 1 - B
            D = torch.log(C)
            per_class_loss[class_number] = D#torch.log(self.eps+1-torch.norm(qbar @ q.T))

        # Place all the class losses into a single list
        losses = [v for v in per_class_loss.values()]

        # Stack the losses to later sum the loss
        stacked_losses = torch.cat(losses)

        # Return the some of all the losses
        return torch.sum(stacked_losses)
        


            





