import argparse
import json
import os
import socket
import subprocess as sb

#TODO: Move these to another module
def set_nccl_env():
    """
    Set DDP related NCCL environment variables
    https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs
    https://github.com/aws/sagemaker-pytorch-training-toolkit/blob/88ca48a831bf4f099d4c57f3c18e0ff92fa2b48c/src/sagemaker_pytorch_container/training.py#L103
    """
    # Disable IB transport and force to use IP sockets by default
    os.environ['NCCL_IB_DISABLE'] = '1'
    # Set to INFO for more NCCL debugging information
    os.environ['NCCL_DEBUG'] = 'WARN'        


def get_sagemaker_resource_config():
    """
    Returns JSON for config if training job is running on SageMaker else None
    """
    cluster_config = None
    sm_config_path = '/opt/ml/input/config/resourceconfig.json'
    if os.path.exists(sm_config_path):
        with open(sm_config_path) as file:
            cluster_config = json.load(file)
    return cluster_config
    

def get_default_hosts():
    cluster_config = get_sagemaker_resource_config()
    if cluster_config:
        hosts = cluster_config['hosts']
        default_nodes = len(hosts)
        default_node_rank = hosts.index(os.environ.get("SM_CURRENT_HOST"))
        
        # elect a leader for PyTorch DDP
        for host in cluster_config['hosts']:
            print(f'host: {host}, IP: {socket.gethostbyname(host)}')
        leader = cluster_config['hosts'][0]  # take first machine in the host list
        
        # Set the network interface for inter node communication
        os.environ['NCCL_SOCKET_IFNAME'] = cluster_config['network_interface_name']
        leader = socket.gethostbyname(hosts[0])
    else:
        # if not on SageMaker, default to single-machine (eg test on Notebook/IDE)
        default_nodes = 1
        default_node_rank = 0
        leader = '127.0.0.1'
    return default_nodes, default_node_rank, leader


def add_training_args(parser):
    """
    This is where we replicate the parameters that our "main" training script requires.
    Additionally, we put parameters that we added to Python SDK estimator `hyperparameter`
    dictionary
    """
    default_nodes, default_node_rank, leader = get_default_hosts()
    
    # Default arguments
    num_epochs_default = 1
    batch_size_default = 256  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "/opt/ml/checkpoints/"
    model_filename_default = "resnet_distributed.pth"
    data_dir_default = "/opt/ml/input/data/dataset"
    
    # model & training parameters
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--use_adascale", action="store_true", help="Use adascale optimizer for training.")
    
    # infra configuration
    parser.add_argument("--gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--nodes", type=int, default=default_nodes)
    parser.add_argument("--node_rank", type=int, default=default_node_rank)
    parser.add_argument("--leader", type=str, default=leader)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--data_dir", type=str, help="Directory for data.", default=data_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)

    return parser


if __name__ == "__main__":    
    set_nccl_env()
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args, _ = parser.parse_known_args()
    print("### Data directory: ", os.listdir(args.data_dir))

    sb.call(['python', '-m',
             # torch cluster config
             'torch.distributed.launch', # DDP launch helper
             '--nproc_per_node', str(args.gpus),
             '--nnodes', str(args.nodes),
             '--node_rank', str(args.node_rank),
             '--master_addr', str(args.leader),
             '--master_port', '7777',

             # training config
             'cifar.py', # main training script
             '--num_epochs',  str(int(args.num_epochs)), \
             '--batch_size', str(int(args.batch_size)) , \
            
             # Data, model, and output directories
             
             '--model_dir', args.model_dir , \
             '--data_dir', args.data_dir 
             
             
            ])
