"""
加载 LLaMA 权重的脚本
将文件2中的 llama_model 的 lm_head 和 embed_tokens 权重加载到文件1中
"""
from dataset.datachuli import load_llama_weights_from_file2

if __name__ == "__main__":
    # ===== 在这里写死两个 .pth 文件路径 =====
    pth_path1 = "/root/autodl-tmp/merged_checkpoint_interactive.pth"  # 修改为第一个 checkpoint 路径
    pth_path2 = "/root/autodl-tmp/LW2L/sava/medsam/二阶段medsam/checkpoints/checkpoint_epoch0_step2169_bleu0.124830_cider0.197326.pth"  # 修改为第二个 checkpoint 路径
    output_path = "/root/autodl-tmp/merged_checkpoint_interactive.pth"  # 修改为输出路径
    
    # 调用函数加载权重
    load_llama_weights_from_file2(
        file1_path=pth_path1,
        file2_path=pth_path2,
        output_path=output_path  # 如果为 None，会覆盖 file1_path
    )




