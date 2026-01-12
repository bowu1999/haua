import copy
import time
import os
import numpy as np
import torch
import torch.nn as nn

# --- PyTorch 相关依赖 ---
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

# --- ONNX 相关依赖 ---
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None
    print("[Warning] onnx 或 onnxruntime 未安装，ONNX 分析功能将不可用。")


class ModelAnalyzer:
    """
    PyTorch 模型结构与性能分析工具
    支持计算 FLOPs (thop/calflops)、参数量、显存占用、推理延迟。
    Usage:
        model = resnet50()
        # 注意：BatchSize 会直接影响显存和 FLOPs
        dummy_input = torch.randn(1, 3, 224, 224)
        # 初始化分析器
        analyzer = ModelAnalyzer(model, dummy_input, device='cuda')
        # 执行分析
        analyzer.getParamsAndFlops() # 计算算力
        analyzer.getMemoryUsage()     # 计算显存
        analyzer.getLatency()          # 计算速度
        # 可选择不同的计算后端
        # 1. 使用 calflops (推荐用于 Transformer/LLM，也支持 CNN)
        analyzer.getParamsAndFlops(method='calflops')
        # 2. 使用 thop (经典 CNN 推荐)
        # analyzer.getParamsAndFlops(method='thop')
    Usage (PyTorch):
        analyzer = ModelAnalyzer(torch_model, dummy_input_tensor)
        analyzer.getParamsAndFlops()
    
    Usage (ONNX):
        # input_sample 可选，如果不传则尝试根据模型定义自动生成
        analyzer = ModelAnalyzer("path/to/model.onnx", device='cuda')
        analyzer.getParamsAndFlops()
    """
    def __init__(self, model, input_sample=None, device='cuda'):
        """
        Args:
            model: PyTorch模型 (nn.Module) 或 ONNX文件路径 (str)
            input_sample: 
                - PyTorch: 必须提供 (Tensor)
                - ONNX: 可选 (numpy array)，若为 None 则自动生成
            device: 'cuda' 或 'cpu'
        """
        self.device_name = device
        self.is_onnx = False
        self.onnx_session = None
        self.onnx_model_proto = None
        
        # 判断模型类型
        if isinstance(model, str) and model.endswith('.onnx'):
            self._init_onnx(model, device)
        elif isinstance(model, nn.Module):
            self._init_pytorch(model, input_sample, device)
        else:
            raise ValueError("model 参数必须是 nn.Module 对象或 .onnx 文件路径")

        # 处理输入样本 (ONNX 模式下如果未提供则自动生成)
        if self.is_onnx:
            if input_sample is None:
                self.input_sample = self._generate_onnx_dummy_input()
            else:
                self.input_sample = input_sample
        else:
            if input_sample is None:
                raise ValueError("PyTorch 模式下必须提供 input_sample (Tensor)")
            self.input_sample = input_sample.to(self.device)

    def _init_pytorch(self, model, input_sample, device):
        """初始化 PyTorch 环境"""
        self.is_onnx = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

    def _init_onnx(self, model_path, device):
        """初始化 ONNX 环境"""
        if onnx is None or ort is None:
            raise ImportError("请安装 onnx 和 onnxruntime: pip install onnx onnxruntime-gpu")
        
        self.is_onnx = True
        self.model_path = model_path
        
        # 1. 加载模型结构用于分析参数
        self.onnx_model_proto = onnx.load(model_path)
        
        # 2. 加载推理 Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        try:
            self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"[Warning] 创建 ONNX Session 失败 (可能缺少 CUDA 库)，回退到 CPU: {e}")
            self.onnx_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        print(f"ONNX 模型已加载: {model_path}")
        print(f"运行设备: {self.onnx_session.get_providers()[0]}")

    def _generate_onnx_dummy_input(self):
        """根据 ONNX 输入节点自动生成随机 Numpy 数据"""
        inputs = {}
        print("正在根据 ONNX 签名自动生成 Dummy Input...")
        for node in self.onnx_session.get_inputs(): # type: ignore
            name = node.name
            shape = node.shape
            elem_type = node.type
            
            # 处理动态维度 (例如 batch size 为 'None' 或 字符串)
            new_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim is None:
                    new_shape.append(1) # 默认 Batch Size = 1
                else:
                    new_shape.append(dim)
            
            # 生成数据
            if 'float' in elem_type.lower():
                data = np.random.randn(*new_shape).astype(np.float32)
            elif 'int' in elem_type.lower() or 'long' in elem_type.lower():
                data = np.random.randint(0, 10, size=tuple(new_shape)).astype(np.int64)
            elif 'bool' in elem_type.lower():
                data = np.random.choice([True, False], size=tuple(new_shape))
            else:
                # 默认 float32
                data = np.random.randn(*new_shape).astype(np.float32)
            
            inputs[name] = data
        return inputs

    def getParamsAndFlops(self, method='calflops'):
        """计算参数量 (ONNX 暂不支持精确 FLOPs，仅统计参数)"""
        print("-" * 30)
        
        if self.is_onnx:
            print("正在分析 ONNX 模型参数...")
            # ONNX 计算 FLOPs 比较复杂，通常需要专门的工具(如 onnx-tool)，这里主要统计参数量
            # 统计 Initializers (权重)
            param_count = 0
            for initializer in self.onnx_model_proto.graph.initializer: # type: ignore
                # 计算每个权重的元素个数
                dims = initializer.dims
                if len(dims) > 0:
                    param_count += np.prod(dims)
            
            # 统计文件大小作为参考
            file_size = os.path.getsize(self.model_path) / (1024 * 1024)
            
            print(f"Parameters (参数量): {param_count / 1e6:.2f} M")
            print(f"Model File Size (文件大小): {file_size:.2f} MB")
            print("Note: ONNX 模式下暂未集成 FLOPs 计算 (推荐使用 onnx-tool 库进行深度分析)")
            return param_count, 0

        else:
            # --- PyTorch 逻辑保持不变 ---
            print(f"正在计算 FLOPs 和参数量 (使用后端: {method})...")
            model_clone = copy.deepcopy(self.model)
            model_clone.eval()
            params = 0.0
            flops = 0.0
            macs = 0.0

            try:
                if method == 'thop':
                    if profile is None: raise ImportError("请安装 thop")
                    macs, params = profile(model_clone, inputs=(self.input_sample, ), verbose=False) # type: ignore
                    flops = macs * 2
                elif method == 'calflops':
                    if calculate_flops is None: raise ImportError("请安装 calflops")
                    flops, macs, params = calculate_flops(
                        model=model_clone,
                        input_shape=tuple(self.input_sample.shape), # type: ignore
                        output_as_string=False, print_results=False, print_detailed=False
                    )
                else:
                    raise ValueError(f"不支持的方法: {method}")
            except Exception as e:
                print(f"[Error] 计算 FLOPs 失败: {e}")
                return 0, 0
            finally:
                del model_clone
                if self.device.type == 'cuda': torch.cuda.empty_cache()

            print(f"Parameters (参数量): {params / 1e6:.2f} M") # type: ignore
            print(f"FLOPs (浮点运算): {flops / 1e9:.2f} G") # type: ignore
            return params, flops

    def getMemoryUsage(self):
        """计算显存/内存占用 (支持 PyTorch 和 ONNX)"""
        print("-" * 30)
        print("正在测试显存占用...")
        
        mem_used = 0
        
        # === 1. ONNX 模式处理 ===
        if self.is_onnx:
            if self.device_name == 'cuda' and torch.cuda.is_available():
                import gc
                try:
                    # 1. 清理环境，准备测量
                    if self.onnx_session is not None:
                        # 必须先删除当前的 session，否则显存已经被占用了，测不出增量
                        del self.onnx_session
                        self.onnx_session = None
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # 2. 获取基准显存 (Baseline)
                    # mem_get_info 返回 (free, total)，单位是字节
                    free_base, total_mem = torch.cuda.mem_get_info()
                    base_used = total_mem - free_base
                    
                    # 3. 重新加载 Session (模拟模型加载过程)
                    print(" -> 正在重新加载 Session 以测量显存增量...")
                    providers = ['CUDAExecutionProvider']
                    # 注意：这里重新创建 session
                    session = ort.InferenceSession(self.model_path, providers=providers) # type: ignore
                    
                    # 4. 运行一次推理 (触发 Activation/Arena 显存分配)
                    input_feed = self.input_sample
                    output_names = [x.name for x in session.get_outputs()]
                    session.run(output_names, input_feed)
                    
                    # 5. 获取峰值显存 (Peak)
                    free_peak, _ = torch.cuda.mem_get_info()
                    peak_used = total_mem - free_peak
                    
                    # 6. 计算差值
                    mem_used = (peak_used - base_used) / (1024 ** 2) # 转换为 MB
                    
                    # 恢复 session 给后续方法使用
                    self.onnx_session = session
                    
                    print(f"Estimated GPU Memory (预估显存占用): {mem_used:.2f} MB")
                    print("   (注: 该数值为全局显存增量，包含权重+推理上下文)")
                    
                except Exception as e:
                    print(f"[Error] ONNX 显存测量失败: {e}")
                    # 尝试恢复 session 以免影响后续调用
                    if self.onnx_session is None:
                        self._init_onnx(self.model_path, self.device_name)
            else:
                print("[Info] 当前为 CPU 模式，无法精确测量 ONNX 内存占用。")
                
        # === 2. PyTorch 模式处理 ===
        else:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.no_grad():
                        _ = self.model(self.input_sample)
                    
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
        
        if self.is_onnx:
            # --- ONNX 推理 ---
            input_feed = self.input_sample # 字典格式 {name: numpy_array}
            output_names = [x.name for x in self.onnx_session.get_outputs()] # type: ignore
            
            # 预热
            print("预热中...")
            try:
                for _ in range(10):
                    self.onnx_session.run(output_names, input_feed) # type: ignore
            except Exception as e:
                print(f"[Error] ONNX 推理失败: {e}")
                return 0

            # 测速
            # 注意: Python 端测速包含了一些 Python 开销，但对于端到端评估是合理的
            for _ in range(repetitions):
                start_time = time.time()
                self.onnx_session.run(output_names, input_feed) # type: ignore
                end_time = time.time()
                timings.append((end_time - start_time) * 1000) # ms
            
            # 计算吞吐量 (假设第一个输入的第一个维度是 Batch Size)
            first_input = list(input_feed.values())[0]
            batch_size = first_input.shape[0]

        else:
            # --- PyTorch 推理 ---
            # 预热
            print("预热中...")
            try:
                with torch.no_grad():
                    for _ in range(10): _ = self.model(self.input_sample)
            except Exception as e:
                print(f"[Error] 预热失败: {e}")
                return 0

            # 测速
            with torch.no_grad():
                if self.device.type == 'cuda':
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    for _ in range(repetitions):
                        starter.record() # type: ignore
                        _ = self.model(self.input_sample)
                        ender.record() # type: ignore
                        torch.cuda.synchronize()
                        timings.append(starter.elapsed_time(ender))
                else:
                    for _ in range(repetitions):
                        start = time.time()
                        _ = self.model(self.input_sample)
                        timings.append((time.time() - start) * 1000)
            
            batch_size = self.input_sample.shape[0] # type: ignore

        # 统计结果
        avg_latency = np.mean(timings)
        std_latency = np.std(timings)
        throughput = 1000 / avg_latency * batch_size 
        
        print(f"Average Latency (平均延迟): {avg_latency:.2f} ms ± {std_latency:.2f}")
        print(f"Throughput (吞吐量): {throughput:.2f} samples/sec")
        return avg_latency

    def showSummary(self):
        """打印模型结构"""
        print("-" * 30)
        if self.is_onnx:
            print("ONNX 模型摘要:")
            print(f"IR Version: {self.onnx_model_proto.ir_version}") # type: ignore
            print(f"Opset Version: {self.onnx_model_proto.opset_import[0].version}") # type: ignore
            print("Inputs:")
            for inp in self.onnx_session.get_inputs(): # type: ignore
                print(f" - Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
            print("Outputs:")
            for out in self.onnx_session.get_outputs(): # type: ignore
                print(f" - Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
        else:
            if summary is None:
                print("torchinfo 未安装。")
                return
            try:
                summary(self.model, input_data=self.input_sample)
            except Exception as e:
                print(f"摘要生成失败: {e}")


if __name__ == "__main__":
    # 1. PyTorch 模式示例
    print("=== PyTorch Mode ===")
    pt_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    )
    pt_input = torch.randn(1, 3, 224, 224)
    analyzer_pt = ModelAnalyzer(pt_model, pt_input, device='cpu') # 或 cuda
    analyzer_pt.getParamsAndFlops()
    analyzer_pt.getLatency()

    # 2. ONNX 模式示例 (假设你有一个 model.onnx 文件)
    # 为了演示，我们先导出一个 onnx 文件
    torch.onnx.export(pt_model, pt_input, "test_model.onnx", 
                      input_names=['input'], output_names=['output'])
    
    print("\n=== ONNX Mode ===")
    # 注意：这里不需要手动传 input_sample，它会自动生成
    analyzer_onnx = ModelAnalyzer("test_model.onnx", device='cpu') 
    analyzer_onnx.showSummary()
    analyzer_onnx.getParamsAndFlops()
    analyzer_onnx.getLatency()
    
    # 清理临时文件
    if os.path.exists("test_model.onnx"):
        os.remove("test_model.onnx")