import copy
import time
import inspect  # 新增：用于分析模型签名
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# --- 可选依赖 ---
try:
    from torchinfo import summary
except ImportError:
    summary = None

try:
    from thop import profile # type: ignore
except ImportError:
    profile = None

try:
    from calflops import calculate_flops # type: ignore
except ImportError:
    calculate_flops = None


class ModelAnalyzer:
    """
    PyTorch 模型结构与性能分析工具 (智能参数适配版)
    """
    def __init__(self, model, input_sample, device='cuda'):
        """
        Args:
            model: PyTorch模型 (nn.Module)
            input_sample: 输入样本
            device: 'cuda' 或 'cpu'
        """
        if not isinstance(model, nn.Module):
            raise ValueError("model 参数必须是 nn.Module 对象")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

        # 1. 处理输入数据移动
        self.input_sample = self._move_to_device(input_sample, self.device)
        
        # 2. 智能分析 forward 签名，决定是否解包输入
        self.use_unpacking = self._check_unpacking_strategy()

    def _move_to_device(self, data, device):
        """递归将数据移动到指定设备"""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (tuple, list)):
            return type(data)(self._move_to_device(x, device) for x in data)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        else:
            return data

    def _check_unpacking_strategy(self):
        """
        判断推理时是否需要对 Tuple/List 输入进行解包 (*args)。
        解决 forward(feats_tuple) 和 forward(x1, x2) 的冲突。
        """
        # 如果输入不是列表或元组，肯定不需要解包
        if not isinstance(self.input_sample, (tuple, list)):
            return False

        try:
            # 获取 forward 函数的签名
            sig = inspect.signature(self.model.forward)
            # 获取所有参数（不包含 self）
            params = list(sig.parameters.values())
            
            # 过滤掉 *args 和 **kwargs，只看显式的位置参数
            positional_params = [
                p for p in params 
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            
            num_model_args = len(positional_params)
            num_input_items = len(self.input_sample)

            # 核心逻辑：
            # 如果模型只需要 1 个位置参数，但输入包含 >1 个元素
            # 说明这一个参数本身就是一个 Tuple，因此【不应该】解包。
            if num_model_args == 1 and num_input_items > 1:
                print(
                    f"[Info] 检测到模型仅接收 1 个参数，输入为包含 {num_input_items} 个元素的 Tuple"
                    f"将保持 Tuple 整体传入，不进行解包。")
                return False
            
            # 其他情况默认解包 (例如模型需要 2 个参数，输入也是 2 个元素)
            return True
            
        except Exception as e:
            print(f"[Warning] 自动分析 forward 签名失败: {e}。默认使用解包模式。")
            return True

    def _get_input_batch_size(self):
        """尝试从输入中获取 Batch Size"""
        if isinstance(self.input_sample, torch.Tensor):
            return self.input_sample.shape[0]
        elif isinstance(self.input_sample, (tuple, list)) and len(self.input_sample) > 0:
            # 递归检查第一个元素
            first_item = self.input_sample[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.shape[0]
        return 1

    def _forward_pass(self, model, inputs):
        """统一的推理调用接口"""
        if self.use_unpacking and isinstance(inputs, (tuple, list)):
            return model(*inputs) # 解包：model(t1, t2)
        else:
            return model(inputs)  # 不解包：model((t1, t2))

    def getParamsAndFlops(self, method='thop'):
        """计算参数量和 FLOPs"""
        print("-" * 30)
        print(f"正在计算 FLOPs 和参数量 (使用后端: {method})...")
        
        model_clone = copy.deepcopy(self.model)
        model_clone.eval()
        
        params = 0.0
        flops = 0.0
        macs = 0.0

        try:
            if method == 'thop':
                if profile is None: raise ImportError("请安装 thop: pip install thop")
                
                # thop 的 inputs 参数必须是一个 Tuple，代表参数列表
                if self.use_unpacking:
                    # 如果需要解包，thop 接收 (t1, t2, t3)
                    thop_inputs = self.input_sample if isinstance(self.input_sample, tuple) \
                        else tuple(self.input_sample)
                else:
                    # 如果不需要解包（整体传入），thop 接收 ((t1, t2, t3), ) -> 一个包含Tuple的Tuple
                    thop_inputs = (self.input_sample, )

                macs, params = profile(model_clone, inputs=thop_inputs, verbose=False) # type: ignore
                flops = macs * 2
                
            elif method == 'calflops':
                if calculate_flops is None: 
                    raise ImportError("请安装 calflops: pip install calflops")
                
                # calflops 需要 input_shape
                if isinstance(self.input_sample, (tuple, list)):
                    # 简单处理：获取所有 tensor 的 shape
                    input_shape = tuple(
                        x.shape for x in self.input_sample if isinstance(x, torch.Tensor))
                else:
                    input_shape = tuple(self.input_sample.shape)
                
                flops, macs, params = calculate_flops(
                    model=model_clone,
                    input_shape=input_shape, # type: ignore
                    output_as_string=False, print_results=False, print_detailed=False)
            else:
                raise ValueError(f"不支持的方法: {method}")
                
        except Exception as e:
            print(f"[Error] 计算 FLOPs 失败: {e}")
            params = sum(p.numel() for p in model_clone.parameters())
            print(f"[Info] 仅统计参数量。")
        finally:
            del model_clone
            if self.device.type == 'cuda': torch.cuda.empty_cache()

        print(f"Parameters (参数量): {params / 1e6:.2f} M") # type: ignore
        if flops > 0:
            print(f"FLOPs (浮点运算): {flops / 1e9:.2f} G") # type: ignore
        return params, flops

    def getMemoryUsage(self):
        """计算显存占用 (仅支持 CUDA)"""
        print("-" * 30)
        print("正在测试显存占用...")
        
        mem_used = 0
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    _ = self._forward_pass(self.model, self.input_sample)
                
                mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
                print(f"Peak GPU Memory (峰值显存): {mem_used:.2f} MB")
            except Exception as e:
                print(f"[Error] 显存测试失败: {e}")
        else:
            print("当前使用 CPU，无法精确测量峰值显存 (仅支持 CUDA)")
            mem_used = 0

        return mem_used

    def getLatency(self, repetitions=100):
        """计算推理延迟"""
        print("-" * 30)
        print(f"正在测试推理速度 (循环 {repetitions} 次)...")
        
        timings = []
        
        # 1. 预热
        print("预热中...")
        try:
            with torch.no_grad():
                for _ in range(10): 
                    _ = self._forward_pass(self.model, self.input_sample)
        except Exception as e:
            print(f"[Error] 预热失败: {e}")
            return 0

        # 2. 测速
        with torch.no_grad():
            if self.device.type == 'cuda':
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                for _ in range(repetitions):
                    starter.record() # type: ignore
                    _ = self._forward_pass(self.model, self.input_sample)
                    ender.record() # type: ignore
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender)) # ms
            else:
                for _ in range(repetitions):
                    start = time.time()
                    _ = self._forward_pass(self.model, self.input_sample)
                    timings.append((time.time() - start) * 1000) # ms
        
        batch_size = self._get_input_batch_size()

        # 统计结果
        avg_latency = np.mean(timings)
        std_latency = np.std(timings)
        throughput = 1000 / avg_latency * batch_size 
        
        print(f"Average Latency (平均延迟): {avg_latency:.2f} ms ± {std_latency:.2f}")
        print(f"Throughput (吞吐量): {throughput:.2f} samples/sec (BatchSize={batch_size})")
        return avg_latency

    def showSummary(self):
        """打印模型结构"""
        print("-" * 30)
        if summary is None:
            print("torchinfo 未安装。")
            return
        try:
            # torchinfo 处理 input_data 比较灵活，通常直接传即可
            # 但为了保险，如果 use_unpacking 为 False，我们将其包装为 list
            input_data = self.input_sample
            if not self.use_unpacking:
                input_data = [self.input_sample] # 包装成 [ (t1, t2, t3) ]
            
            summary(self.model, input_data=input_data)
        except Exception as e:
            print(f"摘要生成失败: {e}")

# --- 你的模型定义 (保持不变) ---
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleModule(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return torch.nn.functional.interpolate(self.conv(x), scale_factor=2)

class PrototypeSegmenter(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int]):
        super().__init__()
        self.proto = nn.Sequential(
            ConvBNAct(in_channels[0], 64, kernel_size=3, stride=1),
            UpsampleModule(64, 64),
            ConvBNAct(64, 64, kernel_size=3, stride=1),
            ConvBNAct(64, 32, kernel_size=3, stride=1))
        self.module_list = nn.ModuleList()
        for i_c in in_channels:
            self.module_list.append(
                nn.Sequential(
                    ConvBNAct(i_c, 32, kernel_size=3, stride=1),
                    ConvBNAct(32, 32, kernel_size=3, stride=1),
                    nn.Conv2d(32, 32, kernel_size=1, stride=1)))
    
    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        prototype_mask = self.proto(feats[0])
        outs = ()
        for i, module in enumerate(self.module_list):
            feat = module(feats[i])       
            B, C, H, W = feat.shape
            feat = feat.view(B, C, -1)   
            outs += (feat,)

        return prototype_mask, outs

# --- 测试代码 ---
if __name__ == "__main__":
    print("初始化模型...")
    model = PrototypeSegmenter([512, 512, 768])
    
    # 构造输入 (Tuple)
    dummy_input = (
        torch.randn(2, 512, 80, 80), 
        torch.randn(2, 512, 40, 40), 
        torch.randn(2, 768, 20, 20)
    )
    
    # 初始化分析器
    # 此时会自动检测到 PrototypeSegmenter.forward 只接受 1 个参数，
    # 因此会自动将 dummy_input 作为一个整体传入，而不会拆开。
    analyzer = ModelAnalyzer(model, dummy_input, device='cuda')
    
    # 执行分析
    analyzer.getParamsAndFlops() 
    analyzer.getMemoryUsage()     
    analyzer.getLatency()