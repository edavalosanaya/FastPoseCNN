
import numpy as np

import torch
import torch.nn

import catalyst.core.callback
import catalyst.core.callbacks

import pdb

# Local Imports
import visualize

#-------------------------------------------------------------------------------

class MyCustomCallback(catalyst.core.callback.Callback):
    def __init__(self):
        super().__init__(order=catalyst.core.callback.CallbackOrder.Logging, 
                         node=catalyst.core.callback.CallbackNode.Master,
                         scope=catalyst.core.callback.CallbackScope.Experiment)

    def on_batch_end(self, runner):

        print('1')

class TensorAddImageCallback(catalyst.core.callbacks.logging.TensorboardLogger):

    # https://catalyst-team.github.io/catalyst/api/core.html#catalyst.core.callbacks.logging.TensorboardLogger

    def __init__(self):
        super().__init__(metric_names=['summary image'],
                         log_on_batch_end=True,
                         log_on_epoch_end=False)

    def on_batch_end(self, runner):

        pdb.set_trace()

        # Utilize the best loaders
        loader_names = list(runner.loaders.keys())

        if 'test' in loader_names:
            best_loader = 'test'
        elif 'valid' in loader_names:
            best_loader = 'valid'
        else:
            best_loader = 'train'

        # Get random sample
        sample = next(iter(runner.loaders[best_loader]))

        # Create the summary figure
        summary_fig = self.mask_check_tb(sample, runner)

        # Log the figure to tensorboard
        self.loggers['_base'].add_figure(f'summary', summary_fig, runner.global_sample_step)

    def mask_check_tb(self, sample, runner):

        # Decompose the sample (in tensor) to numpy
        gt_mask = sample['mask'].cpu().numpy()
        image_vis = sample['image'].cpu().numpy().astype(np.uint8)
        
        # Given the sample, make the prediction with the runner
        logits = runner.predict_batch(sample)['logits']
        pr_mask = torch.nn.functional.sigmoid(logits).cpu().numpy()

        if pr_mask.shape[1] == 1: # Binary segmentation
            pr_mask = pr_mask[:,0,:,:]
            gt_mask_vis = gt_mask[:,0,:,:]

        else: # Multi-class segmentation
            pr_mask = np.argmax(pr_mask, axis=1)
            gt_mask_vis = np.argmax(gt_mask, axis=1)

        # Obtaining colormap given the dataset

        # Creating a matplotlib figure illustrating the inputs vs outputs
        summary_fig = visualize.make_summary_figure(
            image=image_vis,
            ground_truth_mask=gt_mask_vis,
            predicited_mask=pr_mask)

        return summary_fig