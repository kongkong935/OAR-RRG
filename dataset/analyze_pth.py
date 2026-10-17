"""
分析PyTorch模型文件(.pth)的工具
输入pth文件路径，输出文件内的键、结构以及模块维度
"""
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict


def analyze_pth_file(pth_path: str) -> Dict[str, Any]:
    """
    分析pth文件的结构和内容
    
    Args:
        pth_path: pth文件的路径
        
    Returns:
        包含键、结构和维度信息的字典
    """
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"文件不存在: {pth_path}")
    
    print(f"正在加载文件: {pth_path}")
    # 使用 weights_only=False 以支持加载包含 Lightning 等自定义对象的 checkpoint
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    
    result = {
        'file_path': str(pth_path),
        'top_level_keys': [],
        'structure': {},
        'modules': {},
        'total_params': 0
    }
    
    # 分析顶层键
    if isinstance(checkpoint, dict):
        result['top_level_keys'] = list(checkpoint.keys())
        print(f"\n顶层键: {result['top_level_keys']}")
        
        # 尝试提取state_dict
        state_dict = None
        for key in ["state_dict", "model_state_dict", "model", "module", "net", "params"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                print(f"\n找到state_dict在键: '{key}'")
                break
        
        # 如果没有找到常见的state_dict键，检查是否所有键都是字符串（可能是直接的state_dict）
        if state_dict is None:
            if all(isinstance(k, str) for k in checkpoint.keys()):
                state_dict = checkpoint
                print("\n整个checkpoint就是state_dict")
        
        if state_dict is not None:
            result['structure'], result['modules'], result['total_params'] = analyze_state_dict(state_dict)
        else:
            # 分析整个checkpoint的结构
            result['structure'] = analyze_dict_structure(checkpoint)
    else:
        print(f"\n警告: checkpoint不是字典类型，而是: {type(checkpoint)}")
        result['structure'] = {'type': str(type(checkpoint)), 'value': str(checkpoint)}
    
    return result


def analyze_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, int]:
    """
    分析state_dict的结构和维度
    
    Args:
        state_dict: 模型的state_dict
        
    Returns:
        (structure_dict, modules_dict, total_params)
    """
    structure = {}
    modules = defaultdict(list)
    total_params = 0
    
    print(f"\n总共 {len(state_dict)} 个参数")
    print("\n" + "="*80)
    print("参数详情:")
    print("="*80)
    
    for key, tensor in state_dict.items():
        # 提取模块名（第一个点号前的部分）
        if '.' in key:
            module_name = key.split('.')[0]
        else:
            module_name = key
        
        # 记录到对应模块
        modules[module_name].append({
            'key': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': tensor.numel()
        })
        
        # 计算参数数量
        num_params = tensor.numel()
        total_params += num_params
        
        # 构建结构信息
        structure[key] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': num_params,
            'module': module_name
        }
        
        # 打印详细信息
        print(f"{key:60s} | Shape: {str(tensor.shape):30s} | Dtype: {str(tensor.dtype):10s} | Params: {num_params:>12,}")
    
    print("="*80)
    print(f"\n总参数数量: {total_params:,}")
    
    # 按模块汇总
    print("\n" + "="*80)
    print("按模块汇总:")
    print("="*80)
    for module_name in sorted(modules.keys()):
        module_params = sum(item['numel'] for item in modules[module_name])
        print(f"\n模块: {module_name}")
        print(f"  参数数量: {len(modules[module_name])}")
        print(f"  总参数量: {module_params:,}")
        print(f"  参数列表:")
        for item in modules[module_name]:
            print(f"    - {item['key']:50s} | Shape: {str(item['shape']):30s} | Params: {item['numel']:>12,}")
    
    # 转换为普通字典
    modules_dict = {k: v for k, v in modules.items()}
    
    return structure, modules_dict, total_params


def analyze_dict_structure(obj: Any, max_depth: int = 3, current_depth: int = 0) -> Dict:
    """
    递归分析字典结构
    
    Args:
        obj: 要分析的对象
        max_depth: 最大递归深度
        current_depth: 当前深度
        
    Returns:
        结构字典
    """
    if current_depth >= max_depth:
        return {'type': str(type(obj)), 'truncated': True}
    
    if isinstance(obj, dict):
        structure = {}
        for key, value in obj.items():
            if isinstance(value, (dict, list, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    structure[key] = {
                        'type': 'torch.Tensor',
                        'shape': list(value.shape),
                        'dtype': str(value.dtype)
                    }
                elif isinstance(value, dict):
                    structure[key] = analyze_dict_structure(value, max_depth, current_depth + 1)
                else:
                    structure[key] = {'type': str(type(value)), 'length': len(value)}
            else:
                structure[key] = {'type': str(type(value)), 'value': str(value)[:100]}
        return structure
    elif isinstance(obj, list):
        if len(obj) > 0:
            return {
                'type': 'list',
                'length': len(obj),
                'first_item': analyze_dict_structure(obj[0], max_depth, current_depth + 1)
            }
        else:
            return {'type': 'list', 'length': 0}
    else:
        return {'type': str(type(obj)), 'value': str(obj)[:100]}


def print_summary(result: Dict[str, Any]):
    """
    打印分析结果摘要
    
    Args:
        result: analyze_pth_file返回的结果
    """
    print("\n" + "="*80)
    print("分析摘要")
    print("="*80)
    print(f"文件路径: {result['file_path']}")
    print(f"顶层键: {result['top_level_keys']}")
    
    if result['modules']:
        print(f"\n模块数量: {len(result['modules'])}")
        print(f"模块列表: {list(result['modules'].keys())}")
        print(f"总参数数量: {result['total_params']:,}")
    
    print("="*80)


def main():
    """
    主函数 - 直接在代码中指定pth文件路径
    """
    # ========== 在这里修改pth文件路径 ==========
    pth_path = "/root/autodl-tmp/LW2L/sava/单qformer-lora-medsam/二阶段medsam/checkpoint/checkpoint_epoch1_step13018_bleu0.119626_cider0.222873.pth"  # 修改为你的pth文件路径
    # ==========================================
    
    try:
        result = analyze_pth_file(pth_path)
        print_summary(result)
        
        # 可选：保存结果到JSON文件
        import json
        output_path = Path(pth_path).with_suffix('.analysis.json')
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_result = convert_to_serializable(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

