
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn

import catalyst.core.callback
import catalyst.core.callbacks

import pdb

# Local Imports
import visualize

#-------------------------------------------------------------------------------

class TensorAddImageCallback(catalyst.core.callbacks.logging.TensorboardLogger):

    # https://catalyst-team.github.io/catalyst/api/core.html#catalyst.core.callbacks.logging.TensorboardLogger

    def __init__(self, colormap):
        super().__init__(metric_names=None,
                         log_on_batch_end=True,
                         log_on_epoch_end=True)

        self.colormap = colormap

    def on_epoch_end(self, runner):
        super().on_epoch_end(runner)

        # Logging images for both 'train' and 'valid'
        for mode in ['train', 'valid']:
            self.shared_epoch_end(mode, runner)

    def shared_epoch_end(self, mode, runner):

        # Get random sample
        sample = next(iter(runner.loaders[mode]))

        # Only using three images
        items_to_use = np.random.choice(np.arange(sample['mask'].shape[0]), 3, replace=False)
        for key in sample.keys():
            sample[key] = sample[key][items_to_use]

        #pdb.set_trace()

        # Create the summary figure
        summary_fig = self.mask_check_tb(sample, runner)

        # Log the figure to tensorboard
        self.loggers[f'{mode}_log'].add_figure(f'mask_gen/{mode}', summary_fig, runner.global_sample_step)

    def mask_check_tb(self, sample, runner):

        #pdb.set_trace()

        # Selecting clean image and mask if available
        if 'clean mask' in sample.keys():
            gt_mask = sample['clean mask'].cpu().numpy()
        else:
            gt_mask = sample['mask']

        if 'clean image' in sample.keys():
            image_vis = sample['clean image'].cpu().numpy().astype(np.uint8)
        else:
            image_vis = sample['image'].cpu().numpy().astype(np.uint8)

        #pdb.set_trace()
        
        # Given the sample, make the prediction with the runner
        logits = runner.predict_batch(sample)['logits']
        pr_mask = torch.nn.functional.sigmoid(logits).cpu().numpy()

        #pdb.set_trace()

        # Target (ground truth) data format 
        if len(gt_mask.shape) == len('BCHW'):

            if pr_mask.shape[1] == 1: # Binary segmentation
                pr_mask = pr_mask[:,0,:,:]
                gt_mask = gt_mask[:,0,:,:]

            else: # Multi-class segmentation
                pr_mask = np.argmax(pr_mask, axis=1)
                gt_mask = np.argmax(gt_mask, axis=1)

        elif len(gt_mask.shape) == len('BHW'):

            if pr_mask.shape[1] == 1: # Binary segmentation
                pr_mask = pr_mask[:,0,:,:]

            else: # Multi-class segmentation
                pr_mask = np.argmax(pr_mask, axis=1)

        # Colorized the binary masks
        #pdb.set_trace()

        gt_mask_vis = visualize.get_visualized_masks(gt_mask, self.colormap)
        pr_mask = visualize.get_visualized_masks(pr_mask, self.colormap)

        # Creating a matplotlib figure illustrating the inputs vs outputs
        summary_fig = visualize.make_summary_figure(
            image=image_vis,
            ground_truth_mask=gt_mask_vis,
            predicited_mask=pr_mask)

        #pdb.set_trace()

        return summary_fig