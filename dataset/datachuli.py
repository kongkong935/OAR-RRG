"""
数据加载和处理相关的工具函数
"""
import torch
from pathlib import Path
from typing import Set, List, Tuple, Dict
from collections import defaultdict


def extract_state_dict(ckpt):
    """
    从 checkpoint 中提取 state_dict
    
    Args:
        ckpt: 加载的 checkpoint 对象
        
    Returns:
        state_dict: 提取出的 state_dict
    """
    if isinstance(ckpt, dict):
        # 尝试常见的 key
        for k in ["state_dict", "model_state_dict", "model", "module", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # 如果所有 key 都是字符串，可能是直接的 state_dict
        if all(isinstance(x, str) for x in ckpt.keys()):
            return ckpt
    raise ValueError("无法从 checkpoint 中提取 state_dict")


def get_modules_from_state_dict(state_dict: Dict) -> Set[str]:
    """
    从 state_dict 中提取所有模块名称
    
    Args:
        state_dict: 模型的 state_dict
        
    Returns:
        Set[str]: 所有模块名称的集合（例如 "qformer_front", "llama_model", "visual_encoder"）
    """
    modules = set()
    for key in state_dict.keys():
        # 提取第一个点号前的部分作为模块名
        # 例如 "qformer_front.layer.0.weight" -> "qformer_front"
        if '.' in key:
            module_name = key.split('.')[0]
            modules.add(module_name)
        else:
            # 如果没有点号，整个 key 就是模块名
            modules.add(key)
    return modules


def compare_common_modules_values(state_dict1: Dict, state_dict2: Dict, common_keys: Set[str]) -> Dict:
    """
    比较共同模块的数值是否相同
    
    Returns:
        Dict: 包含比较结果的字典
    """
    results = {
        "identical_keys": [],      # 完全相同的 key
        "different_keys": [],       # 不同的 key
        "missing_in_file1": [],    # 在文件2中但不在文件1中的 key
        "missing_in_file2": [],    # 在文件1中但不在文件2中的 key
        "shape_mismatch": [],      # shape 不匹配的 key
    }
    
    for key in common_keys:
        if key not in state_dict1:
            results["missing_in_file1"].append(key)
            continue
        if key not in state_dict2:
            results["missing_in_file2"].append(key)
            continue
        
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        # 检查 shape 是否相同
        if tensor1.shape != tensor2.shape:
            results["shape_mismatch"].append((key, tensor1.shape, tensor2.shape))
            continue
        
        # 检查数值是否相同
        if torch.equal(tensor1, tensor2):
            results["identical_keys"].append(key)
        else:
            # 计算差异统计
            diff = torch.abs(tensor1.float() - tensor2.float())
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            results["different_keys"].append({
                "key": key,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "shape": list(tensor1.shape)
            })
    
    return results


def compare_pth_modules(pth_path1: str, pth_path2: str, check_values: bool = True) -> Tuple[Dict, Dict, Dict]:
    """
    比较两个 .pth 文件内部包含的模块，并可选地检查共同模块的数值
    
    Args:
        pth_path1: 第一个 .pth 文件路径
        pth_path2: 第二个 .pth 文件路径
        check_values: 是否检查共同模块的数值是否相同
        
    Returns:
        Tuple[Dict, Dict, Dict]: 
            - 第一个字典：包含两个文件的模块信息
            - 第二个字典：包含比较结果（共同模块、独有模块等）
            - 第三个字典：包含数值比较结果（如果 check_values=True）
    """
    # 加载两个 checkpoint
    print(f"正在加载: {pth_path1}")
    ckpt1 = torch.load(pth_path1, map_location="cpu", weights_only=False)
    state_dict1 = extract_state_dict(ckpt1)
    
    print(f"正在加载: {pth_path2}")
    ckpt2 = torch.load(pth_path2, map_location="cpu", weights_only=False)
    state_dict2 = extract_state_dict(ckpt2)
    
    # 提取模块名称
    modules1 = get_modules_from_state_dict(state_dict1)
    modules2 = get_modules_from_state_dict(state_dict2)
    
    # 统计信息
    info = {
        "file1": {
            "path": pth_path1,
            "total_keys": len(state_dict1),
            "modules": sorted(modules1),
            "module_count": len(modules1)
        },
        "file2": {
            "path": pth_path2,
            "total_keys": len(state_dict2),
            "modules": sorted(modules2),
            "module_count": len(modules2)
        }
    }
    
    # 比较结果
    common_modules = modules1 & modules2  # 交集
    only_in_file1 = modules1 - modules2   # 只在文件1中
    only_in_file2 = modules2 - modules1   # 只在文件2中
    
    comparison = {
        "common_modules": sorted(common_modules),
        "only_in_file1": sorted(only_in_file1),
        "only_in_file2": sorted(only_in_file2),
        "common_count": len(common_modules),
        "only_file1_count": len(only_in_file1),
        "only_file2_count": len(only_in_file2)
    }
    
    # 检查共同模块的数值
    value_comparison = None
    if check_values:
        common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
        print(f"\n正在比较 {len(common_keys)} 个共同参数的数值...")
        value_comparison = compare_common_modules_values(state_dict1, state_dict2, common_keys)
    
    return info, comparison, value_comparison


def print_pth_modules(pth_path1: str, pth_path2: str, check_values: bool = True):
    """
    打印两个 .pth 文件的模块信息和数值比较结果
    
    Args:
        pth_path1: 第一个 .pth 文件路径
        pth_path2: 第二个 .pth 文件路径
        check_values: 是否检查共同模块的数值是否相同
    """
    info, comparison, value_comparison = compare_pth_modules(pth_path1, pth_path2, check_values)
    
    print("\n" + "="*80)
    print("📦 Checkpoint 文件模块分析")
    print("="*80)
    
    # 文件1信息
    print(f"\n📄 文件 1: {Path(info['file1']['path']).name}")
    print(f"   - 总参数键数: {info['file1']['total_keys']}")
    print(f"   - 模块数量: {info['file1']['module_count']}")
    print(f"   - 模块列表:")
    for module in info['file1']['modules']:
        print(f"     • {module}")
    
    # 文件2信息
    print(f"\n📄 文件 2: {Path(info['file2']['path']).name}")
    print(f"   - 总参数键数: {info['file2']['total_keys']}")
    print(f"   - 模块数量: {info['file2']['module_count']}")
    print(f"   - 模块列表:")
    for module in info['file2']['modules']:
        print(f"     • {module}")
    
    # 比较结果
    print(f"\n🔍 模块比较结果:")
    print(f"   - 共同模块 ({comparison['common_count']} 个):")
    if comparison['common_modules']:
        for module in comparison['common_modules']:
            print(f"     ✓ {module}")
    else:
        print("     (无)")
    
    print(f"\n   - 仅在文件1中 ({comparison['only_file1_count']} 个):")
    if comparison['only_in_file1']:
        for module in comparison['only_in_file1']:
            print(f"     + {module}")
    else:
        print("     (无)")
    
    print(f"\n   - 仅在文件2中 ({comparison['only_file2_count']} 个):")
    if comparison['only_in_file2']:
        for module in comparison['only_in_file2']:
            print(f"     + {module}")
    else:
        print("     (无)")
    
    # 数值比较结果
    if value_comparison:
        print(f"\n🔬 共同模块数值比较结果:")
        print(f"   - 完全相同的参数: {len(value_comparison['identical_keys'])} 个")
        if len(value_comparison['identical_keys']) > 0 and len(value_comparison['identical_keys']) <= 10:
            for key in value_comparison['identical_keys'][:10]:
                print(f"     ✓ {key}")
        elif len(value_comparison['identical_keys']) > 10:
            print(f"     (显示前10个)")
            for key in value_comparison['identical_keys'][:10]:
                print(f"     ✓ {key}")
        
        print(f"\n   - 数值不同的参数: {len(value_comparison['different_keys'])} 个")
        if value_comparison['different_keys']:
            print(f"     (显示前10个差异最大的)")
            # 按 max_diff 排序
            sorted_diffs = sorted(value_comparison['different_keys'], key=lambda x: x['max_diff'], reverse=True)
            for item in sorted_diffs[:10]:
                print(f"     ✗ {item['key']}")
                print(f"       最大差异: {item['max_diff']:.6e}, 平均差异: {item['mean_diff']:.6e}")
                print(f"       Shape: {item['shape']}")
        
        if value_comparison['shape_mismatch']:
            print(f"\n   - Shape 不匹配的参数: {len(value_comparison['shape_mismatch'])} 个")
            for key, shape1, shape2 in value_comparison['shape_mismatch'][:10]:
                print(f"     ⚠ {key}")
                print(f"       File1: {shape1}, File2: {shape2}")
        
        if value_comparison['missing_in_file1']:
            print(f"\n   - 在文件2中但不在文件1中的参数: {len(value_comparison['missing_in_file1'])} 个")
            for key in value_comparison['missing_in_file1'][:10]:
                print(f"     - {key}")
        
        if value_comparison['missing_in_file2']:
            print(f"\n   - 在文件1中但不在文件2中的参数: {len(value_comparison['missing_in_file2'])} 个")
            for key in value_comparison['missing_in_file2'][:10]:
                print(f"     - {key}")
    
    print("\n" + "="*80)


def merge_pth_files(pth_path1: str, pth_path2: str, output_path: str, exclude_modules: Set[str] = None):
    """
    合并两个 .pth 文件：
    - 独有的模块：都保存
    - 共有的模块：只保存一份（优先使用文件1的）
    - 排除指定模块（如 llama_model）
    
    Args:
        pth_path1: 第一个 .pth 文件路径
        pth_path2: 第二个 .pth 文件路径
        output_path: 输出合并后的 .pth 文件路径
        exclude_modules: 要排除的模块名称集合（例如 {"llama_model"}）
    """
    if exclude_modules is None:
        exclude_modules = {"llama_model"}
    
    print(f"正在加载: {pth_path1}")
    ckpt1 = torch.load(pth_path1, map_location="cpu", weights_only=False)
    state_dict1 = extract_state_dict(ckpt1)
    
    print(f"正在加载: {pth_path2}")
    ckpt2 = torch.load(pth_path2, map_location="cpu", weights_only=False)
    state_dict2 = extract_state_dict(ckpt2)
    
    # 提取模块名称
    modules1 = get_modules_from_state_dict(state_dict1)
    modules2 = get_modules_from_state_dict(state_dict2)
    
    # 找出共同模块和独有模块
    common_modules = modules1 & modules2
    only_in_file1 = modules1 - modules2
    only_in_file2 = modules2 - modules1
    
    # 构建合并后的 state_dict
    merged_state_dict = {}
    
    # 1. 保存文件1中独有的模块（排除指定模块）
    print(f"\n📦 处理文件1的独有模块 ({len(only_in_file1)} 个)...")
    for key in state_dict1.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in only_in_file1 and module_name not in exclude_modules:
            merged_state_dict[key] = state_dict1[key]
            print(f"   ✓ 添加: {key} (来自文件1)")
    
    # 2. 保存文件2中独有的模块（排除指定模块）
    print(f"\n📦 处理文件2的独有模块 ({len(only_in_file2)} 个)...")
    for key in state_dict2.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in only_in_file2 and module_name not in exclude_modules:
            merged_state_dict[key] = state_dict2[key]
            print(f"   ✓ 添加: {key} (来自文件2)")
    
    # 3. 保存共同模块（只保存一份，优先使用文件1的，排除指定模块）
    print(f"\n📦 处理共同模块 ({len(common_modules)} 个)...")
    for key in state_dict1.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in common_modules and module_name not in exclude_modules:
            # 如果这个 key 在两个文件中都存在，使用文件1的
            if key in state_dict2:
                merged_state_dict[key] = state_dict1[key]
                print(f"   ✓ 添加: {key} (共同模块，使用文件1的值)")
            else:
                # 如果只在文件1中存在，也添加
                merged_state_dict[key] = state_dict1[key]
                print(f"   ✓ 添加: {key} (来自文件1，文件2中无此key)")
    
    # 4. 处理文件2中共同模块的 key，但文件1中没有的
    for key in state_dict2.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in common_modules and module_name not in exclude_modules:
            if key not in merged_state_dict:
                merged_state_dict[key] = state_dict2[key]
                print(f"   ✓ 添加: {key} (共同模块，文件1中无此key，使用文件2的值)")
    
    # 5. 排除指定模块
    excluded_count = 0
    for key in list(merged_state_dict.keys()):
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in exclude_modules:
            del merged_state_dict[key]
            excluded_count += 1
    
    if excluded_count > 0:
        print(f"\n🚫 已排除 {excluded_count} 个属于排除模块的参数")
    
    # 保存合并后的 checkpoint
    # 保留原始 checkpoint 的其他信息（如 epoch, step 等）
    merged_ckpt = {}
    if isinstance(ckpt1, dict):
        for k in ["epoch", "step", "config", "hparams"]:
            if k in ckpt1:
                merged_ckpt[k] = ckpt1[k]
    
    merged_ckpt["model"] = merged_state_dict
    
    print(f"\n💾 保存合并后的 checkpoint 到: {output_path}")
    torch.save(merged_ckpt, output_path)
    
    print(f"\n✅ 合并完成！")
    print(f"   - 合并后的参数总数: {len(merged_state_dict)}")
    print(f"   - 文件1原始参数数: {len(state_dict1)}")
    print(f"   - 文件2原始参数数: {len(state_dict2)}")
    
    return merged_state_dict


def interactive_merge_pth_files(pth_path1: str, pth_path2: str, output_path: str):
    """
    交互式合并两个 .pth 文件，使用图形界面或文本菜单让用户选择要合并的模块
    
    Args:
        pth_path1: 第一个 .pth 文件路径
        pth_path2: 第二个 .pth 文件路径
        output_path: 输出合并后的 .pth 文件路径
    """
    # 检测是否有图形界面
    has_display = False
    try:
        import os
        has_display = 'DISPLAY' in os.environ or os.name == 'nt'
        if has_display:
            import tkinter as tk
            from tkinter import ttk
            import tkinter.messagebox
    except:
        has_display = False
    
    # 加载两个 checkpoint
    print(f"\n正在加载: {pth_path1}")
    ckpt1 = torch.load(pth_path1, map_location="cpu", weights_only=False)
    state_dict1 = extract_state_dict(ckpt1)
    
    print(f"正在加载: {pth_path2}")
    ckpt2 = torch.load(pth_path2, map_location="cpu", weights_only=False)
    state_dict2 = extract_state_dict(ckpt2)
    
    # 提取模块名称
    modules1 = get_modules_from_state_dict(state_dict1)
    modules2 = get_modules_from_state_dict(state_dict2)
    
    # 找出所有模块（共同 + 独有）
    all_modules = sorted(modules1 | modules2)
    common_modules = modules1 & modules2
    only_in_file1 = modules1 - modules2
    only_in_file2 = modules2 - modules1
    
    # 显示模块信息
    print("\n" + "="*80)
    print("📦 两个 Checkpoint 文件的模块信息")
    print("="*80)
    print(f"\n📄 文件 1: {Path(pth_path1).name}")
    print(f"   - 总参数键数: {len(state_dict1)}")
    print(f"   - 模块数量: {len(modules1)}")
    
    print(f"\n📄 文件 2: {Path(pth_path2).name}")
    print(f"   - 总参数键数: {len(state_dict2)}")
    print(f"   - 模块数量: {len(modules2)}")
    
    print(f"\n📋 所有模块列表:")
    print(f"   - 共同模块 ({len(common_modules)} 个): {sorted(common_modules)}")
    print(f"   - 仅在文件1中 ({len(only_in_file1)} 个): {sorted(only_in_file1)}")
    print(f"   - 仅在文件2中 ({len(only_in_file2)} 个): {sorted(only_in_file2)}")
    
    # 根据是否有图形界面选择不同的交互方式
    selected_modules = None
    if has_display:
        try:
            # 尝试使用图形界面
            selected_modules = _gui_select_modules(all_modules, common_modules, only_in_file1, only_in_file2, pth_path1, pth_path2)
        except Exception as e:
            print(f"\n⚠️  图形界面初始化失败 ({e})，切换到文本菜单模式")
            selected_modules = _text_select_modules(all_modules, common_modules, only_in_file1, only_in_file2)
    else:
        # 使用文本菜单
        selected_modules = _text_select_modules(all_modules, common_modules, only_in_file1, only_in_file2)
    
    if not selected_modules:
        print("已取消合并")
        return
    
    # 显示选择结果
    print(f"\n✅ 已选择 {len(selected_modules)} 个模块:")
    for module in sorted(selected_modules):
        print(f"   • {module}")
    
    # 开始合并
    print("\n" + "="*80)
    print("🔀 开始合并选中的模块")
    print("="*80)
    
    merged_state_dict = {}
    
    # 1. 处理文件1中独有的模块
    for key in state_dict1.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in selected_modules and module_name in only_in_file1:
            merged_state_dict[key] = state_dict1[key]
            print(f"   ✓ 添加: {key} (来自文件1)")
    
    # 2. 处理文件2中独有的模块
    for key in state_dict2.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in selected_modules and module_name in only_in_file2:
            merged_state_dict[key] = state_dict2[key]
            print(f"   ✓ 添加: {key} (来自文件2)")
    
    # 3. 处理共同模块（优先使用文件1的值）
    for key in state_dict1.keys():
        module_name = key.split('.')[0] if '.' in key else key
        if module_name in selected_modules and module_name in common_modules:
            merged_state_dict[key] = state_dict1[key]
            print(f"   ✓ 添加: {key} (来自文件1，共同模块)")
    
    # 4. 保存合并后的 checkpoint（确保使用 'model' 键，匹配 weight_utils.py 的期望）
    print(f"\n💾 保存合并后的 checkpoint 到: {output_path}")
    
    # 保持文件1的其他信息（如 epoch, step 等）
    merged_ckpt = {}
    if isinstance(ckpt1, dict):
        for k in ["epoch", "step", "config", "hparams", "optimizer", "lr_scheduler"]:
            if k in ckpt1:
                merged_ckpt[k] = ckpt1[k]
    
    # 始终使用 'model' 键保存 state_dict（匹配 weight_utils.py 的期望）
    merged_ckpt["model"] = merged_state_dict
    
    torch.save(merged_ckpt, output_path)
    print(f"✅ 合并完成！共合并 {len(merged_state_dict)} 个参数键")
    print(f"   - Checkpoint 格式: 包含 'model' 键（匹配 weight_utils.py 的期望）")
    
    return merged_state_dict


def _text_select_modules(all_modules, common_modules, only_in_file1, only_in_file2):
    """使用文本菜单选择模块"""
    print("\n" + "="*80)
    print("🔀 请选择要合并的模块（输入编号，用逗号分隔，如: 1,3,5 或输入 'all' 选择全部）")
    print("="*80)
    
    # 显示模块列表供选择
    module_list = []
    for idx, module in enumerate(all_modules, 1):
        status = ""
        if module in common_modules:
            status = " [共同]"
        elif module in only_in_file1:
            status = " [仅文件1]"
        elif module in only_in_file2:
            status = " [仅文件2]"
        
        print(f"   [{idx:2d}] {module}{status}")
        module_list.append(module)
    
    # 获取用户选择
    while True:
        user_input = input("\n请输入要合并的模块编号（用逗号分隔，如: 1,3,5）或 'all' 选择全部，'q' 退出: ").strip()
        
        if user_input.lower() == 'q':
            return set()
        
        if user_input.lower() == 'all':
            return set(all_modules)
        
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            selected_modules = set()
            for idx in indices:
                if 1 <= idx <= len(module_list):
                    selected_modules.add(module_list[idx - 1])
                else:
                    print(f"⚠️  编号 {idx} 超出范围，已忽略")
            
            if selected_modules:
                return selected_modules
            else:
                print("❌ 没有选择任何模块，请重新输入")
        except ValueError:
            print("❌ 输入格式错误，请输入数字编号（用逗号分隔）或 'all'")


def _gui_select_modules(all_modules, common_modules, only_in_file1, only_in_file2, pth_path1, pth_path2):
    """使用图形界面选择模块"""
    import tkinter as tk
    from tkinter import ttk
    import tkinter.messagebox
    
    selected_modules = set()
    window_closed = False
    
    def on_confirm():
        nonlocal selected_modules
        selected_modules = {all_modules[i] for i, var in enumerate(check_vars) if var.get()}
        if not selected_modules:
            tk.messagebox.showwarning("警告", "请至少选择一个模块！")
            return
        root.quit()
        root.destroy()
    
    def on_cancel():
        nonlocal window_closed
        window_closed = True
        root.quit()
        root.destroy()
    
    def select_all():
        for var in check_vars:
            var.set(True)
    
    def deselect_all():
        for var in check_vars:
            var.set(False)
    
    # 创建主窗口
    root = tk.Tk()
    root.title("选择要合并的模块")
    root.geometry("600x700")
    
    # 创建滚动框架
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # 标题
    title_label = tk.Label(scrollable_frame, text="请选择要合并的模块", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    # 文件信息
    info_frame = tk.Frame(scrollable_frame)
    info_frame.pack(pady=5, padx=10, fill="x")
    
    file1_label = tk.Label(info_frame, text=f"文件1: {Path(pth_path1).name}", font=("Arial", 10))
    file1_label.pack(anchor="w")
    
    file2_label = tk.Label(info_frame, text=f"文件2: {Path(pth_path2).name}", font=("Arial", 10))
    file2_label.pack(anchor="w")
    
    # 全选/全不选按钮
    button_frame = tk.Frame(scrollable_frame)
    button_frame.pack(pady=5)
    
    select_all_btn = tk.Button(button_frame, text="全选", command=select_all, width=10)
    select_all_btn.pack(side="left", padx=5)
    
    deselect_all_btn = tk.Button(button_frame, text="全不选", command=deselect_all, width=10)
    deselect_all_btn.pack(side="left", padx=5)
    
    # 分隔线
    separator = ttk.Separator(scrollable_frame, orient='horizontal')
    separator.pack(fill='x', pady=10)
    
    # 创建复选框
    check_vars = []
    for module in all_modules:
        var = tk.BooleanVar()
        check_vars.append(var)
        
        # 确定状态标签
        if module in common_modules:
            status = " [共同]"
            status_color = "blue"
        elif module in only_in_file1:
            status = " [仅文件1]"
            status_color = "green"
        elif module in only_in_file2:
            status = " [仅文件2]"
            status_color = "orange"
        else:
            status = ""
            status_color = "black"
        
        # 创建复选框框架
        check_frame = tk.Frame(scrollable_frame)
        check_frame.pack(anchor="w", padx=20, pady=2)
        
        check_btn = tk.Checkbutton(
            check_frame,
            text=module,
            variable=var,
            font=("Arial", 10)
        )
        check_btn.pack(side="left")
        
        status_label = tk.Label(
            check_frame,
            text=status,
            fg=status_color,
            font=("Arial", 9)
        )
        status_label.pack(side="left", padx=5)
    
    # 确认和取消按钮
    confirm_frame = tk.Frame(root)
    confirm_frame.pack(pady=10)
    
    confirm_btn = tk.Button(confirm_frame, text="确认合并", command=on_confirm, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=15)
    confirm_btn.pack(side="left", padx=10)
    
    cancel_btn = tk.Button(confirm_frame, text="取消", command=on_cancel, bg="#f44336", fg="white", font=("Arial", 12), width=15)
    cancel_btn.pack(side="left", padx=10)
    
    # 打包滚动条和画布
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 运行窗口
    root.mainloop()
    
    # 检查是否取消
    if window_closed or not selected_modules:
        return set()
    
    return selected_modules


def _text_select_modules(all_modules, common_modules, only_in_file1, only_in_file2):
    """使用文本菜单选择模块"""
    print("\n" + "="*80)
    print("🔀 请选择要合并的模块（输入编号，用逗号分隔，如: 1,3,5 或输入 'all' 选择全部）")
    print("="*80)
    
    # 显示模块列表供选择
    module_list = []
    for idx, module in enumerate(all_modules, 1):
        status = ""
        if module in common_modules:
            status = " [共同]"
        elif module in only_in_file1:
            status = " [仅文件1]"
        elif module in only_in_file2:
            status = " [仅文件2]"
        
        print(f"   [{idx:2d}] {module}{status}")
        module_list.append(module)
    
    # 获取用户选择
    while True:
        user_input = input("\n请输入要合并的模块编号（用逗号分隔，如: 1,3,5）或 'all' 选择全部，'q' 退出: ").strip()
        
        if user_input.lower() == 'q':
            return set()
        
        if user_input.lower() == 'all':
            return set(all_modules)
        
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            selected_modules = set()
            for idx in indices:
                if 1 <= idx <= len(module_list):
                    selected_modules.add(module_list[idx - 1])
                else:
                    print(f"⚠️  编号 {idx} 超出范围，已忽略")
            
            if selected_modules:
                return selected_modules
            else:
                print("❌ 没有选择任何模块，请重新输入")
        except ValueError:
            print("❌ 输入格式错误，请输入数字编号（用逗号分隔）或 'all'")


def _gui_select_modules(all_modules, common_modules, only_in_file1, only_in_file2, pth_path1, pth_path2):
    """使用图形界面选择模块"""
    import tkinter as tk
    from tkinter import ttk
    import tkinter.messagebox
    
    selected_modules = set()
    window_closed = False
    
    def on_confirm():
        nonlocal selected_modules
        selected_modules = {all_modules[i] for i, var in enumerate(check_vars) if var.get()}
        if not selected_modules:
            tk.messagebox.showwarning("警告", "请至少选择一个模块！")
            return
        root.quit()
        root.destroy()
    
    def on_cancel():
        nonlocal window_closed
        window_closed = True
        root.quit()
        root.destroy()
    
    def select_all():
        for var in check_vars:
            var.set(True)
    
    def deselect_all():
        for var in check_vars:
            var.set(False)
    
    # 创建主窗口
    root = tk.Tk()
    root.title("选择要合并的模块")
    root.geometry("600x700")
    
    # 创建滚动框架
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # 标题
    title_label = tk.Label(scrollable_frame, text="请选择要合并的模块", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    # 文件信息
    info_frame = tk.Frame(scrollable_frame)
    info_frame.pack(pady=5, padx=10, fill="x")
    
    file1_label = tk.Label(info_frame, text=f"文件1: {Path(pth_path1).name}", font=("Arial", 10))
    file1_label.pack(anchor="w")
    
    file2_label = tk.Label(info_frame, text=f"文件2: {Path(pth_path2).name}", font=("Arial", 10))
    file2_label.pack(anchor="w")
    
    # 全选/全不选按钮
    button_frame = tk.Frame(scrollable_frame)
    button_frame.pack(pady=5)
    
    select_all_btn = tk.Button(button_frame, text="全选", command=select_all, width=10)
    select_all_btn.pack(side="left", padx=5)
    
    deselect_all_btn = tk.Button(button_frame, text="全不选", command=deselect_all, width=10)
    deselect_all_btn.pack(side="left", padx=5)
    
    # 分隔线
    separator = ttk.Separator(scrollable_frame, orient='horizontal')
    separator.pack(fill='x', pady=10)
    
    # 创建复选框
    check_vars = []
    for module in all_modules:
        var = tk.BooleanVar()
        check_vars.append(var)
        
        # 确定状态标签
        if module in common_modules:
            status = " [共同]"
            status_color = "blue"
        elif module in only_in_file1:
            status = " [仅文件1]"
            status_color = "green"
        elif module in only_in_file2:
            status = " [仅文件2]"
            status_color = "orange"
        else:
            status = ""
            status_color = "black"
        
        # 创建复选框框架
        check_frame = tk.Frame(scrollable_frame)
        check_frame.pack(anchor="w", padx=20, pady=2)
        
        check_btn = tk.Checkbutton(
            check_frame,
            text=module,
            variable=var,
            font=("Arial", 10)
        )
        check_btn.pack(side="left")
        
        status_label = tk.Label(
            check_frame,
            text=status,
            fg=status_color,
            font=("Arial", 9)
        )
        status_label.pack(side="left", padx=5)
    
    # 确认和取消按钮
    confirm_frame = tk.Frame(root)
    confirm_frame.pack(pady=10)
    
    confirm_btn = tk.Button(confirm_frame, text="确认合并", command=on_confirm, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=15)
    confirm_btn.pack(side="left", padx=10)
    
    cancel_btn = tk.Button(confirm_frame, text="取消", command=on_cancel, bg="#f44336", fg="white", font=("Arial", 12), width=15)
    cancel_btn.pack(side="left", padx=10)
    
    # 打包滚动条和画布
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 运行窗口
    root.mainloop()
    
    # 检查是否取消
    if window_closed or not selected_modules:
        return set()
    
    return selected_modules


def load_llama_weights_from_file2(
    file1_path: str,
    file2_path: str,
    output_path: str = None
):
    """
    将文件2中的 llama_model 的 lm_head 和 embed_tokens 权重加载到文件1中
    
    Args:
        file1_path: 文件1（目标文件）的路径
        file2_path: 文件2（源文件）的路径，从中提取 llama 权重
        output_path: 输出文件路径，如果为 None 则覆盖文件1
    
    Returns:
        dict: 更新后的 state_dict
    """
    # 加载文件1
    print(f"📂 加载文件1（目标）: {file1_path}")
    ckpt1 = torch.load(file1_path, map_location="cpu", weights_only=False)
    state_dict1 = extract_state_dict(ckpt1)
    
    # 加载文件2
    print(f"📂 加载文件2（源）: {file2_path}")
    ckpt2 = torch.load(file2_path, map_location="cpu", weights_only=False)
    state_dict2 = extract_state_dict(ckpt2)
    
    # 要复制的权重键
    target_keys = [
        "llama_model.base_model.model.lm_head.weight",
        "llama_model.base_model.model.model.embed_tokens.weight"
    ]
    
    # 复制权重
    updated_count = 0
    not_found_count = 0
    
    for key in target_keys:
        if key in state_dict2:
            if key in state_dict1:
                # 检查形状是否匹配
                if state_dict1[key].shape == state_dict2[key].shape:
                    state_dict1[key] = state_dict2[key]
                    print(f"   ✓ 已更新: {key} (形状: {state_dict1[key].shape})")
                    updated_count += 1
                else:
                    print(f"   ⚠️  形状不匹配，跳过: {key}")
                    print(f"      文件1形状: {state_dict1[key].shape}")
                    print(f"      文件2形状: {state_dict2[key].shape}")
            else:
                # 如果文件1中没有这个键，直接添加
                state_dict1[key] = state_dict2[key]
                print(f"   ✓ 已添加: {key} (形状: {state_dict2[key].shape})")
                updated_count += 1
        else:
            print(f"   ❌ 文件2中未找到: {key}")
            not_found_count += 1
    
    if not_found_count > 0:
        print(f"\n⚠️  警告: 有 {not_found_count} 个权重在文件2中未找到")
    
    # 保存结果
    if output_path is None:
        output_path = file1_path
        print(f"\n⚠️  未指定输出路径，将覆盖文件1: {output_path}")
    
    # 保持文件1的其他信息，并确保使用 'model' 键（匹配 weight_utils.py 的期望）
    if isinstance(ckpt1, dict):
        merged_ckpt = {}
        # 保留文件1的其他信息
        for k in ["epoch", "step", "config", "hparams", "optimizer", "lr_scheduler"]:
            if k in ckpt1:
                merged_ckpt[k] = ckpt1[k]
        
        # 始终使用 'model' 键保存 state_dict（匹配 weight_utils.py 的期望）
        merged_ckpt["model"] = state_dict1
    else:
        # 如果文件1不是字典格式，创建一个包含 'model' 键的字典
        merged_ckpt = {"model": state_dict1}
    
    print(f"\n💾 保存更新后的 checkpoint 到: {output_path}")
    torch.save(merged_ckpt, output_path)
    print(f"✅ 完成！已更新 {updated_count} 个权重")
    print(f"   - Checkpoint 格式: 包含 'model' 键（匹配 weight_utils.py 的期望）")
    
    return state_dict1


def add_weights_to_checkpoint(
    checkpoint_path: str,
    weights_to_add,
    output_path: str = None,
    mode: str = "update"
):
    """
    将指定的权重添加到 checkpoint 中
    
    Args:
        checkpoint_path: 原始 checkpoint 文件路径
        weights_to_add: 要添加的权重，可以是：
            - 字典 (dict): 直接的权重字典，如 {"module.weight": tensor, ...}
            - 字符串 (str): 权重文件路径，会加载该文件并提取 state_dict
        output_path: 输出 checkpoint 文件路径，如果为 None 则覆盖原文件
        mode: 合并模式
            - "update": 更新/覆盖匹配的键（默认）
            - "add": 只添加新键，不覆盖已存在的键
            - "replace": 完全替换，只保留 weights_to_add 中的键
    
    Returns:
        dict: 合并后的 state_dict
    """
    # 加载原始 checkpoint
    print(f"📂 加载原始 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original_state_dict = extract_state_dict(ckpt)
    
    # 处理要添加的权重
    if isinstance(weights_to_add, str):
        # 如果是文件路径，加载它
        print(f"📂 加载权重文件: {weights_to_add}")
        weights_ckpt = torch.load(weights_to_add, map_location="cpu", weights_only=False)
        weights_dict = extract_state_dict(weights_ckpt)
    elif isinstance(weights_to_add, dict):
        # 如果直接是字典，直接使用
        weights_dict = weights_to_add
    else:
        raise ValueError(f"weights_to_add 必须是字典或文件路径字符串，当前类型: {type(weights_to_add)}")
    
    print(f"   - 原始 checkpoint 参数数: {len(original_state_dict)}")
    print(f"   - 要添加的权重参数数: {len(weights_dict)}")
    
    # 根据模式合并权重
    if mode == "update":
        # 更新模式：覆盖匹配的键，保留原有的其他键
        merged_state_dict = original_state_dict.copy()
        updated_count = 0
        new_count = 0
        for key, value in weights_dict.items():
            if key in merged_state_dict:
                merged_state_dict[key] = value
                updated_count += 1
            else:
                merged_state_dict[key] = value
                new_count += 1
        print(f"   - 更新了 {updated_count} 个已存在的参数")
        print(f"   - 添加了 {new_count} 个新参数")
        
    elif mode == "add":
        # 添加模式：只添加新键，不覆盖已存在的
        merged_state_dict = original_state_dict.copy()
        added_count = 0
        skipped_count = 0
        for key, value in weights_dict.items():
            if key not in merged_state_dict:
                merged_state_dict[key] = value
                added_count += 1
            else:
                skipped_count += 1
        print(f"   - 添加了 {added_count} 个新参数")
        print(f"   - 跳过了 {skipped_count} 个已存在的参数（未覆盖）")
        
    elif mode == "replace":
        # 替换模式：完全替换，只保留 weights_to_add 中的键
        merged_state_dict = weights_dict.copy()
        print(f"   - 完全替换，最终参数数: {len(merged_state_dict)}")
        
    else:
        raise ValueError(f"不支持的 mode: {mode}，可选值: 'update', 'add', 'replace'")
    
    # 保存结果
    if output_path is None:
        output_path = checkpoint_path
        print(f"⚠️  未指定输出路径，将覆盖原文件: {output_path}")
    
    # 保持原始 checkpoint 的其他信息（如 epoch, step 等）
    if isinstance(ckpt, dict):
        merged_ckpt = {}
        for k in ["epoch", "step", "config", "hparams", "optimizer", "lr_scheduler"]:
            if k in ckpt:
                merged_ckpt[k] = ckpt[k]
        
        # 找到原始 state_dict 的 key
        state_dict_key = None
        for k in ["state_dict", "model_state_dict", "model", "module", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict_key = k
                break
        
        if state_dict_key:
            merged_ckpt[state_dict_key] = merged_state_dict
        else:
            # 如果找不到，直接使用 "model"
            merged_ckpt["model"] = merged_state_dict
    else:
        merged_ckpt = merged_state_dict
    
    print(f"\n💾 保存合并后的 checkpoint 到: {output_path}")
    torch.save(merged_ckpt, output_path)
    print(f"✅ 完成！最终参数数: {len(merged_state_dict)}")
    
    return merged_state_dict


if __name__ == "__main__":
    # ===== 在这里写死两个 .pth 文件路径 =====
    pth_path1 = "/root/autodl-tmp/LW2L/sava/单qformer-lora-medsam/二阶段medsam/checkpoint/checkpoint_epoch1_step13018_bleu0.119626_cider0.222873.pth"  # 修改为第一个 checkpoint 路径
    pth_path2 = "/root/autodl-tmp/checkpoint_epoch3_step15186_bleu0.129370_cider0.281209.pth"  # 修改为第二个 checkpoint 路径
    output_path = "/root/autodl-tmp/test.pth"  # 修改为输出路径
    
    # 交互式合并
    interactive_merge_pth_files(pth_path1, pth_path2, output_path)

