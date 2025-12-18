import shutil
from pathlib import Path


def move_files_recursive_flat(
    src_dir,
    dst_dir,
    extensions,
    rename_on_conflict=True,
    dry_run=False
):
    """
    递归扫描 src_dir 下所有子目录，将指定后缀的文件移动到 dst_dir（扁平化）。
    
    参数:
        src_dir (str): 源目录路径
        dst_dir (str): 目标目录路径
        extensions (list of str): 文件后缀列表，如 ['.log', 'txt']
        rename_on_conflict (bool): 是否在目标文件存在时自动重命名（默认 True）
        dry_run (bool): 若为 True，仅打印操作而不实际移动（用于测试）
    """
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()

    if not src.exists() or not src.is_dir():
        raise ValueError(f"❌ 源目录不存在: {src}")

    # 创建目标目录
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # 标准化扩展名
    normalized_exts = set()
    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_exts.add(ext)

    # 第一步：收集所有匹配的文件（安全！不在遍历时修改目录）
    matched_files = []
    for file_path in src.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in normalized_exts:
            matched_files.append(file_path)

    if not matched_files:
        print("🔍 未找到任何匹配的文件。")
        return

    print(f"📁 找到 {len(matched_files)} 个匹配文件，开始处理...\n")

    moved_count = 0
    skipped_count = 0

    # 第二步：逐个移动
    for file_path in matched_files:
        target_path = dst / file_path.name

        # 处理冲突：重命名 or 跳过
        original_target = target_path
        if target_path.exists():
            if rename_on_conflict:
                stem = original_target.stem
                suffix = original_target.suffix
                counter = 1
                while target_path.exists():
                    target_path = dst / f"{stem}_{counter}{suffix}"
                    counter += 1
                if not dry_run:
                    print(f"📝 重命名: {file_path.name} → {target_path.name}")
            else:
                print(f"⏭️  跳过（已存在且未启用重命名）: {file_path}")
                skipped_count += 1
                continue
        else:
            if not dry_run:
                print(f"➡️  移动: {file_path} → {target_path}")

        # 执行移动（或 dry run）
        try:
            if not dry_run:
                shutil.move(str(file_path), str(target_path))
            moved_count += 1
        except Exception as e:
            print(f"❌ 移动失败 {file_path}: {e}")

    action = "模拟操作" if dry_run else "实际移动"
    print(f"\n✅ {action}完成！")
    print(f"   - 成功: {moved_count}")
    print(f"   - 跳过: {skipped_count}")
    print(f"   - 目标目录: {dst}")


if __name__ == "__main__":
    # 🔧 请根据实际情况修改以下参数
    source_directory = "/your/source/folder"          # 源目录（多层嵌套）
    destination_directory = "/your/destination/folder"  # 所有文件将移入此目录（扁平化）
    file_extensions = ['.log', '.txt', 'csv']         # 支持多种写法

    # 新增参数：
    #   rename_on_conflict=True  → 默认重命名（解决冲突）
    #   dry_run=True             → 先测试，不真移动（强烈建议首次运行时开启！）
    
    move_files_recursive_flat(
        src_dir=source_directory,
        dst_dir=destination_directory,
        extensions=file_extensions,
        rename_on_conflict=True,   # ← 默认重命名（你要求的）
        dry_run=False              # ← 首次可设为 True 测试
    )