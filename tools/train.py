import os
import torch
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train/Test script for HWP model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    # 初始化分布式环境
    init_dist('pytorch')
    # 设置每个进程使用的GPU
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # 加载配置
    config_file_path = args.config
    print(f"Using config file: {config_file_path}")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    config = Config.fromfile(config_file_path)
    # 模型参数设置
    if not hasattr(config, 'model_wrapper_cfg'):
        config.model_wrapper_cfg = dict(
            type = 'MMDistributedDataParallel',
            find_unused_parameters = True,
            broadcast_buffers = False)
    # 确保launcher设置正确
    config.launcher = 'pytorch'
    # 创建Runner并训练
    runner = Runner.from_cfg(config)
    runner.train()


if __name__ == '__main__':
    main()