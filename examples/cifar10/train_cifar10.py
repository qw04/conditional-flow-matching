# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
import copy
import os

import torch
from absl import app, flags
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange

from utils_cifar import *

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  ## TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 800001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("img_size", 32, help="image size")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  ##Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float(
    "ema_decay", 0.9999, help="ema decay rate"
)
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    5000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "eval_step", 0, help="frequency of evaluating model, 0 to disable during training"
)
flags.DEFINE_integer(
    "num_images", 50000, help="the number of generated images for evaluation"
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print("lr, total_steps, ema decay, save_step:", 
          FLAGS.lr, FLAGS.total_steps, FLAGS.ema_decay, FLAGS.save_step)
    
    #### DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    #### MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    net_node = NeuralODE(
        net_model, solver="euler", sensitivity="adjoint"
    )
    ema_node = NeuralODE(
        ema_model, solver="euler", sensitivity="adjoint"
    )
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    savedir = "./results/"
    os.makedirs(savedir, exist_ok=True)

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    # FM = ConditionalFlowMatcher(sigma=sigma)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    # FM = TargetConditionalFlowMatcher(sigma=sigma)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip
            )  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_node, net_model, 
                                 savedir, step, net_="normal")
                generate_samples(ema_node, ema_model, 
                                 savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + "cifar10_weights_step_{}.pt".format(step),
                )


if __name__ == "__main__":
    app.run(train)
