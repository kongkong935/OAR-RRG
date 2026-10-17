"""
权重加载和冻结相关的工具函数
"""
import torch
import re
from typing import List, Set, Tuple, Optional


def load_checkpoint_and_get_frozen_modules(
    model,
    delta_file: str,
    is_resume: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[Set[str], List[str]]:
    """
    加载checkpoint并返回需要冻结的模块列表（不执行冻结操作）
    
    Args:
        model: 模型实例
        delta_file: checkpoint文件路径
        is_resume: 是否为resume模式（True: 返回空列表, False: 返回需要冻结的模块列表）
        device: 加载设备，如果为None则使用当前CUDA设备
    
    Returns:
        Tuple[Set[str], List[str]]: 
            - 第一个元素：所有需要冻结的模块名称集合（完整列表）
            - 第二个元素：去重后的模块名称列表（用于显示，相同模式只保留第一个）
    """
    if device is None:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    
    # 加载checkpoint
    state_obj = torch.load(delta_file, map_location=device)
    state_dict = state_obj['model']
    model.load_state_dict(state_dict=state_dict, strict=False)
    
    # 如果 resume=True，不需要冻结，返回空列表
    if is_resume:
        return set(), []
    
    # 收集需要冻结的模块（resume=False 时，加载的参数需要冻结）
    loaded_param_names = set(state_dict.keys())
    frozen_modules = set()  # 所有需要冻结的模块
    frozen_modules_pattern = set()  # 用于去重的模式集合
    frozen_modules_list = []  # 保持顺序的去重列表
    
    for name, param in model.named_parameters():
        if name in loaded_param_names and param.requires_grad:
            # 提取模块名称（去掉参数名，只保留模块路径）
            module_name = '.'.join(name.split('.')[:-1])
            if module_name:
                frozen_modules.add(module_name)
                # 将数字索引替换为 *，用于去重（例如 layer.0 -> layer.*）
                module_pattern = re.sub(r'\.\d+\.', '.*.', module_name)
                module_pattern = re.sub(r'\.\d+$', '.*', module_pattern)
                # 只保留第一个出现的模式
                if module_pattern not in frozen_modules_pattern:
                    frozen_modules_pattern.add(module_pattern)
                    frozen_modules_list.append(module_name)
    
    return frozen_modules, frozen_modules_list


def freeze_modules_by_names(model, module_names: Set[str]) -> None:
    """
    根据模块名称列表冻结模型中的模块
    
    Args:
        model: 模型实例
        module_names: 需要冻结的模块名称集合
    """
    for name, param in model.named_parameters():
        # 检查参数是否属于需要冻结的模块
        module_name = '.'.join(name.split('.')[:-1])
        if module_name in module_names and param.requires_grad:
            param.requires_grad = False


def print_frozen_modules(frozen_modules: Set[str], frozen_modules_list: List[str] = None) -> None:
    """
    打印被冻结的模块列表
    
    Args:
        frozen_modules: 所有被冻结的模块名称集合
        frozen_modules_list: 去重后的模块名称列表（用于显示，可选，已废弃）
    """
    if frozen_modules:
        # 提取第一级模块名称（第一个点之前的部分，如果没有点则使用完整名称）
        first_level_modules = set()
        for module_name in frozen_modules:
            if '.' in module_name:
                first_level = module_name.split('.')[0]
            else:
                first_level = module_name
            first_level_modules.add(first_level)
        
        print(f"\n{'='*60}")
        print(f"冻结的模块列表 (共 {len(frozen_modules)} 个模块，第一级模块 {len(first_level_modules)} 个):")
        print(f"{'='*60}")
        for module_name in sorted(first_level_modules):
            print(f"  - {module_name}")
        print(f"{'='*60}\n")
    else:
        print("\n未冻结任何模块（resume=True 或所有参数已冻结）\n")

