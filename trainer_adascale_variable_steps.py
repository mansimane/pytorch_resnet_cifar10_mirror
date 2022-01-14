# Source: https://leimao.github.io/blog/PyTorch-Distributed-Training/
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
from fairscale.optim import AdaScale
import time
import statistics
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import resnet

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))



def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    num_epochs_default = 1
    eval_freq_default = 10
    batch_size_default = 256  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"
    adascale_scale_default = 1
    weight_decaye_default = 1e-4
    momentum_default = 0.9
    base_batch_size = 128.0
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet20)')

    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--eval_freq", type=int, help="Number of epochs for eval.", default=eval_freq_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--weight_decay", type=float, help="Weight decay.", default=weight_decaye_default)
    parser.add_argument("--momentum", type=float, help="Momentum.", default=momentum_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--use_adascale", action="store_true", help="Use adascale optimizer for training.")
    parser.add_argument("--scale_lr_schedule", action="store_true", help="Scale learnining rate schedule based on adascale gain steps",default=False)
    parser.add_argument("--adascale_scale", type=int, help="Scale factor for adascale.", default=adascale_scale_default)

    parser.add_argument("--use_fp16_compress", action="store_true", help="Use fp16 compression for training.")
    parser.add_argument("--run_max_steps", action="store_true", help="Run adascale for number of steps equal to base \
     schedule irrespetive of number of steps", default=True)
    parser.add_argument("--skip_gain_calc_steps", type=int, help="Number of steps for which gain calculation is calculated", default=1)
    parser.add_argument('--log_dir',
                        default='./logs',
                        type=str,
                        help='log directory path.')

    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume
    use_adascale = argv.use_adascale
    use_fp16_compress = argv.use_fp16_compress
    weight_decay = argv.weight_decay
    momentum = argv.momentum
    eval_freq = argv.eval_freq
    run_max_steps = argv.run_max_steps
    scale_lr_schedule = argv.scale_lr_schedule
    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''



    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend="gloo")
    scale = batch_size*get_world_size()/base_batch_size

    if get_rank() == 0:
        # tensorboard summary writer (by default created for all workers)
        tensorboard_path = f'{argv.log_dir}/var-len-worker-0-scale-{scale}-lr-{learning_rate}-bs-{batch_size}-scheduler--adascale-{use_adascale}-shuffle-run_max_steps-{run_max_steps}-scale_lr_schedule-{scale_lr_schedule}'

        writer = SummaryWriter(tensorboard_path)

        print(" Adascale hyperparameters")
        print(" Base batch size: ", batch_size)
        print(" Scale factor batch size: ", batch_size*get_world_size()/base_batch_size)
        print(" Batch size after scaling: ", batch_size*get_world_size())
        print(" Number of steps : ")

    # Encapsulate the model on the GPU assigned to the current process
    model = resnet.__dict__[argv.arch]()

    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if use_fp16_compress:
        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="/home/ubuntu/data/", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="/home/ubuntu/data/", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True)
    print("INFO: Length of dataloader: ", len(train_loader))
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=128,
                             shuffle=False, num_workers=4,
                             pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    if use_adascale:
        print(" INFO: Using Adascale ")
        optimizer = AdaScale(optim.SGD(ddp_model.parameters(),
                                       lr=learning_rate,
                                       momentum=momentum,
                                       weight_decay=weight_decay),
                             scale=scale)
    else:
        print(" INFO: Not using Adascale")
        optimizer = optim.SGD(ddp_model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    step_times = []
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=-1)

    # # Loop over the dataset multiple times
    step = 0
    step_scale_dep = 0

    done = False
    epoch = 0
    last_epoch = 0
    epoch_scaled = 0
    while not done:
        if local_rank == 0:
            print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
            if use_adascale:
                print("Last epoch: ", last_epoch)


        # In distributed mode, calling the :meth:`set_epoch` method at
        #         the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        #         is necessary to make shuffling work properly across multiple epochs. Otherwise,
        #         the same ordering will be always used.

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        train_sampler.set_epoch(epoch)
        # switch to train mode
        ddp_model.train()
        
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            losses.update(loss.item(), inputs.size(0))
            if use_adascale:
                gain = optimizer.gain()
                step_scale_dep += gain

            if get_rank() == 0:
                writer.add_scalar(f'Train/Loss_step', losses.avg, step_scale_dep)
                if use_adascale:
                    writer.add_scalar(f'Gain', gain, step)
                    writer.add_scalar(f'Gain_step_scale_dep', gain, step_scale_dep)
                    writer.add_scalar(f'Train/Loss_step_scale_dep', losses.avg, step_scale_dep)
                    print("Step: ", step,"step_scale_dep", step_scale_dep, "epoch_scaled: ", step_scale_dep // len(train_loader))
                writer.flush()

            optimizer.step()
            step += 1
            if use_adascale and scale_lr_schedule:
                epoch_scaled = step_scale_dep // len(train_loader)
                if epoch_scaled > last_epoch:
                    lr_scheduler.step()
                    last_epoch = epoch_scaled
                    if get_rank() == 0:
                        learning_rate = optimizer.param_groups[0]['lr']
                        writer.add_scalar(f'Learning Rate', learning_rate, epoch_scaled)
                        writer.add_scalar(f'Train/Loss_epoch', losses.avg, epoch)
                        if use_adascale and scale_lr_schedule:
                            writer.add_scalar(f'Train/epoch_scaled', epoch_scaled, epoch)
                        writer.flush()

                    # Save and evaluate model routinely
                    if epoch_scaled % eval_freq == 0:
                        if local_rank == 0:
                            accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                            if get_rank() == 0:
                                writer.add_scalar(f'Val/Acc', accuracy, epoch_scaled)
                                writer.flush()
                            torch.save(ddp_model.state_dict(), model_filepath)
                            print("-" * 75)
                            print("Epoch: {}, Accuracy: {}".format(epoch_scaled, accuracy))
                            print("-" * 75)

            if int(epoch_scaled) >= num_epochs:
                done = True
                break

        epoch += 1
        # Take lr step after every epoch if adascale is not enabled.
        if (not use_adascale) or (not scale_lr_schedule):
            lr_scheduler.step()
    print(" INFO: Total steps: ", step)
    print(" INFO: Total step_scale_dep: ", step_scale_dep)



if __name__ == "__main__":
    main()
