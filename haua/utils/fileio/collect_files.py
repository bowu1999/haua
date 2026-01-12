from pathlib import Path
from typing import List, Union, Iterable


def get_files_by_suffix(
    root: Union[str, Path],
    suffixes: Union[str, Iterable[str]],
    recursive: bool = True,
    return_str: bool = False,
) -> List[Union[Path, str]]:
    """
    获取指定路径下指定后缀的所有文件

    Args:
        root: 目标目录
        suffixes: 文件后缀，如 ".jpg" 或 [".jpg", ".png"]
        recursive: 是否递归搜索子目录
        return_str: 是否返回字符串路径（默认返回 Path）

    Returns:
        文件路径列表
    """
    root = Path(root)

    if isinstance(suffixes, str):
        suffixes = [suffixes]

    # 统一为小写，保证大小写不敏感
    suffixes = {s.lower() for s in suffixes}

    if recursive:
        files = root.rglob("*")
    else:
        files = root.glob("*")

    results = [f for f in files if f.is_file() and f.suffix.lower() in suffixes]

    if return_str:
        results = [str(f) for f in results]

    return results # type: ignore
