import math
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *


savedir = "models/random-moons"
os.makedirs(savedir, exist_ok=True)

def sample_random(n):
    return torch.randn(n, 2)

def learn(l, synthModel, path, training_iteration, show = False):
    sigma = 0.1
    dim = 2
    batch_size = 1000
    model = MLP(dim=dim, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)


    for k in tqdm(range(5000), disable=True):
        optimizer.zero_grad()
    
        x0 = sample_random(batch_size)
        if synthModel == None: 
            x1 = sample_moons(batch_size)
        else:
            with torch.no_grad():
                sn = synthModel.trajectory(sample_random(batch_size * l), t_span=torch.linspace(0, 1, 100))
                sn = sn[-1, :batch_size * l, :]
            x1 = torch.cat((sample_moons(batch_size - len(sn)), sn))
        
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(torch.cat([xt, t[:,None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

        if k % 5000 == 4999 and show:        
            node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            with torch.no_grad():
                traj = node.trajectory(sample_random(2000), t_span=torch.linspace(0, 1, 100))
                temp_path = rf"{path}\\iter_{training_iteration}_k_{k}.png"
                plot_trajectories(traj.cpu().numpy(), temp_path)

    return node
    

def first_iterations(l=0, show=True):
    synthModel = learn(0, None, True)

def many_iterations(l=1, training_iterations=10, showfreq=1):
    path = r"iterative\\"
    if not os.path.exists(path): os.makedirs(path)

    print("Training Iteration 0")
    synth = learn(l, None, path, 0, show=True)
    for t in tqdm(range(training_iterations)):
        print(f"Training iteration {t+1}")
        synth = learn(l, synth, path, t, show = t % showfreq == 0)

def main():
    first_iterations()

if __name__ == "__main__":
    main()