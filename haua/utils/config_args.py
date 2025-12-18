import os
import argparse
import importlib.util
import json
from typing import Any, Dict
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigParser:
    """
    支持 .py, .json, .yaml, .yml 配置文件的解析器

    示例：
        >>> config_parser = ConfigParser()
        >>> args = config_parser.get_args()
    """

    def __init__(self, default_config: str = 'cfg.yaml'):
        self.args = self._parse_args(default_config)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        根据文件扩展名加载配置文件，统一返回字典

        Args:
            config_path (str): 配置文件路径

        Returns:
            Dict[str, Any]: 配置键值对
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")

        _, ext = os.path.splitext(config_path)
        ext = ext.lower()

        if ext == '.py':
            return self._load_py_config(config_path)
        elif ext == '.json':
            return self._load_json_config(config_path)
        elif ext in ('.yaml', '.yml'):
            if not HAS_YAML:
                raise ImportError("请安装 PyYAML: pip install PyYAML")
            return self._load_yaml_config(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}. 支持: .py, .json, .yaml, .yml")

    def _load_py_config(self, config_path: str) -> Dict[str, Any]:
        """加载 .py 配置文件"""
        module_name = "config_module"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        config = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(config)  # type: ignore

        # 提取非魔法属性
        config_dict = {}
        for key in dir(config):
            if not (key.startswith('__') and key.endswith('__')):
                value = getattr(config, key)
                config_dict[key] = value
        return config_dict

    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """加载 .json 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """加载 .yaml / .yml 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _parse_args(self, default_config: str) -> argparse.Namespace:
        """
        解析命令行参数并加载配置文件内容到 args
        """
        parser = argparse.ArgumentParser(description="加载配置文件并解析参数")
        parser.add_argument('-c', '--config', type=str, default=default_config, help='配置文件路径')
        args = parser.parse_args()

        config_dict = self._load_config(args.config)

        # 将配置项添加到 args（转为小写以保持一致性）
        for key, value in config_dict.items():
            setattr(args, key.lower(), value)

        return args

    def get_args(self) -> argparse.Namespace:
        return self.args