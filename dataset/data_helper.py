import os
import json
import re

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor

MASK_LOSS_SIZE = (256, 256)  # (H, W)
class FieldParser:
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.split = split
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model,use_fast=True)
        if getattr(args, 'use_sam', False):
            self.sam_pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            self.sam_pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
            self.sam_target_size = 1024  # SAM的标准输入尺寸
            # self.sam_target_size = 256  # SAM的标准输入尺寸
            self.mask_base_dir = getattr(args, 'mask_dir')  # mask文件根目录


    def resize_mask_for_loss(mask: torch.Tensor, target_size=MASK_LOSS_SIZE) -> torch.Tensor:
        Ht, Wt = target_size
        if mask.ndim == 2:
            if mask.shape[-2:] == (Ht, Wt):
                return mask
            return F.interpolate(mask[None, None], size=(Ht, Wt), mode="nearest").squeeze(0).squeeze(0)

        elif mask.ndim == 3:
            if mask.shape[-2:] == (Ht, Wt):
                return mask
            # 把 Z 视作 batch 维：[Z, H, W] -> [Z, 1, H, W] -> 插值 -> [Z, Ht, Wt]
            return F.interpolate(mask.unsqueeze(1), size=(Ht, Wt), mode="nearest").squeeze(1)

    def _load_mask(self, mask_path):
        full_path = os.path.join(self.mask_base_dir, mask_path)
        arr = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(arr).to(torch.float32)
        mask = (mask > 0).to(torch.float32)  # [H,W]，值域 0/1
        # mask_resized = resize_mask_for_loss(mask, MASK_LOSS_SIZE)
        return mask
    def _parse_image_for_sam(self, img):
        """
        SAM 图像预处理：将 numpy 数组转换为 SAM 所需的张量格式
        """
        H, W = img.shape[:2]
        original_size = (H, W)
        scale = self.sam_target_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((new_W, new_H), Image.BILINEAR)
        img_resized = np.array(img_resized, dtype=np.float32)
        resize_size = (new_H, new_W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)  # [C, H, W]
        img_tensor = (img_tensor - self.sam_pixel_mean) / self.sam_pixel_std
        pad_h = self.sam_target_size - new_H
        pad_w = self.sam_target_size - new_W
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))  # [C, 1024, 1024]
        return img_tensor, (original_size, resize_size)

    # def _parse_image(self, img):
    #     pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
    #     return pixel_values[0]

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # ===== 第1步：保护 [SEG_*] 标记，避免被清洗掉 =====
        # 找到所有 [SEG_xxx] 标记，用占位符替换（不区分大小写）
        seg_pattern = re.compile(r'\[SEG_[^\]]+\]', re.IGNORECASE)
        seg_tokens = seg_pattern.findall(report)

        # 用占位符替换（使用不太可能出现在文本中的标记）
        protected_report = report
        seg_placeholders = []
        for i, seg_token in enumerate(seg_tokens):
            placeholder = f"<SEGTAG{i}>"
            # 保存原始token（保持大小写）
            seg_placeholders.append((placeholder, seg_token))
            protected_report = protected_report.replace(seg_token, placeholder, 1)

        # ===== 第2步：执行原有的清洗逻辑 =====
        # IU X-Ray 数据集清洗规则
        if self.dataset == "iu_xray":
            # 基础清洗：处理重复标点、编号等
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
                .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
                .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():\-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(protected_report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'

        # MIMIC-CXR 数据集清洗规则（更复杂）
        else:
            # 基础清洗：处理换行、下划线、空格等
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                            .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(protected_report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'

        # ===== 第3步：恢复 [SEG_*] 标记（保持原始大小写） =====
        for placeholder, original_token in seg_placeholders:
            report = report.replace(placeholder.lower(), ' ' + original_token + ' ')
        report = re.sub(r'\s+', ' ', report).strip()
        return report


    def parse(self, features):
        to_return = {'id': features['id']}
        to_return['count'] = features.get("count")
        # 训练集和 sam split 优先使用带分割标注的报告；其它集使用原 report
        # to_return['input_text'] = self.clean_report(features.get("report"))
        if self.split.lower() in ['train', 'sam']:
            to_return['input_text'] = self.clean_report(features.get("report_seg"))
        else:
            to_return['input_text'] = self.clean_report(features.get("report"))
        images = []
        for image_path in features['image_path']:
            image = Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB")
            inputs = self.vit_feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            images.append(inputs)
        to_return["image"] = images
        # 单图像时复制一份以保持batch维度一致，在encode_img中通过位置嵌入和count信息处理语义
        if(features.get("count")==1):
            images.append(images[0])

        # 3) SAM图像处理
        if getattr(self.args, 'use_sam', False) and self.split.lower() in ['train', 'test']:
            # 读取第一张图片用于SAM
            first_image_path = features['image_path'][0]
            with Image.open(os.path.join(self.args.base_dir, first_image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)

                # SAM预处理 -> [C, 1024, 1024]
                image_sam, (original_size, resize_size) = self._parse_image_for_sam(array)
                to_return["image_sam"] = image_sam
                to_return["sam_original_size"] = original_size
                to_return["sam_resize_size"] = resize_size

            # 4) MAKS图像处理
            seg_organs = features.get('seg_organs', [])
            seg_mask_paths = features.get('seg_mask_paths', [])

            if seg_organs:
                masks = []
                for mask_path in seg_mask_paths:
                    mask = self._load_mask(mask_path)
                    masks.append(mask)

                to_return["masks"] = masks if masks else []  # 列表: [tensor1, tensor2, ...]
                to_return["seg_organs"] = seg_organs
            else:
                # 没有分割标注
                to_return["masks"] = []
                to_return["seg_organs"] = []

        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args,split=split)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    sam_dataset = ParseDataset(args, 'sam')
    return train_dataset, dev_dataset, test_dataset, sam_dataset



