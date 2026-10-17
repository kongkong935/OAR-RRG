import torch
from test_sam import SAMInterface


def main():
	# 初始化SAM
	sam = SAMInterface("sam_vit_h_4b8939.pth", device="auto")
	
	# 输入：图像路径 + 查询向量
	image_paths = ["0.png", "1.png"]
	query_embeddings = [
		torch.randn(3, 256),  # 图像1的查询向量
		torch.randn(2, 256),  # 图像2的查询向量
	]
	
	# 推理
	final_masks, iou_pred, original_sizes = sam.batch_infer_from_image_paths(
		image_paths=image_paths,
		query_embeddings=query_embeddings,
		batch_size=2,
	)
	
	# 输出：保存掩码（自动处理多查询向量）
	sam.save_masks(final_masks, original_sizes, save_prefix="mask")
	
	print("完成！")


if __name__ == "__main__":
	main()
