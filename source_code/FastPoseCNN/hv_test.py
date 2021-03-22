import tqdm
import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt 
import skimage
import skimage.io
import pprint

# Local Imports 
import tools
import lib
import lib.ransac_voting_gpu_layer.ransac_voting_gpu as rvg

mask = np.load('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/tests_output/0-raw_mask.npy')
hv_img = np.load('/home/students/edavalos/GitHub/FastPoseCNN/source_code/FastPoseCNN/tests_output/0-raw_hv.npy')

pytorch_mask = torch.from_numpy(mask).to('cuda:0')
pytorch_hv = torch.unsqueeze(torch.from_numpy(hv_img).to('cuda:0').permute(0,2,3,1), dim=3)

runtimes = {}

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for N in [10, 20, 50, 75, 100, 128, 200, 300, 400, 500, 10, 20, 50, 75, 100, 128, 200, 300, 400]:

    runtimes[N] = []

    for i in tqdm.tqdm(range(100)):

        start.record()
        
        output = rvg.ransac_voting_layer_v3(
            mask = pytorch_mask, # [b,h,w]
            vertex = pytorch_hv, # [b,h,w,vn,2]
            round_hyp_num = N
        )

        end.record()

        torch.cuda.synchronize()

        runtimes[N].append(start.elapsed_time(end))

    runtimes[N] = np.mean(np.array(runtimes[N]))

pprint.pprint(runtimes)