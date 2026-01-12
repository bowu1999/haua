from typing import Any, Dict

import os
import json


def readJson(filepath: str) -> Any:
    """
    从指定路径读取 JSON 文件并返回解析后的 Python 对象。
    
    Args:
        filepath (str): JSON 文件路径
    
    Returns:
        Any: 解析后的 Python 对象
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def readJsonl(file_path):
    """
    读取JSONL文件（每行一个JSON对象）

    Args:
        file_path (str): 文件路径
    Returns:
        list: JSON对象列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行的JSON格式错误，跳过该行。错误信息：{e}")
                continue

    return data


def saveDictList2jsonlFile(data_list, file_path, ensure_ascii=False, indent=None):
    """
    将字典列表保存为JSONL（JSON Lines）格式的文件

    Args:
        data_list (list): 字典列表，每个元素应为一个字典（可序列化为JSON的对象）
        file_path (str): 保存的文件路径（如'output/test.jsonl'）
        ensure_ascii (bool, optional): 是否确保ASCII编码，默认为False（支持中文等非ASCII字符正常显示）
        indent (int, optional): 缩进的空格数，默认为None（不缩进）
    """
    if not isinstance(data_list, list):
        print(f"错误：输入的data_list不是列表类型，实际类型为{type(data_list)}")
        return False
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"目录{dir_path}不存在，已自动创建")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(data_list):
                try:
                    json_str = json.dumps(
                        item,
                        ensure_ascii=ensure_ascii,
                        indent=indent,
                        sort_keys=False)
                    f.write(json_str + '\n') # type: ignore
                except TypeError as e:
                    print(f"警告：第{idx}个元素无法序列化为JSON，跳过该元素。错误信息：{e}")
                    continue
        print(f"成功将{len(data_list)}条数据保存到{file_path}")
        return True
    except Exception as e:
        print(f"错误：保存文件时发生异常，异常信息：{e}")
        return False


def saveDict2jsonFile(data: Dict[str, Any], filepath: str, indent: int = 4) -> None:
    """
    将字典保存为 JSON 文件。

    :param data: 要保存的字典
    :param filepath: 目标 JSON 文件路径，例如 "data.json"
    :param indent: 缩进空格数，用于美化输出；如果不需要缩进可设为 None
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # 保留中文
            indent=indent)        # 美化格式