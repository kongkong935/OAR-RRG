import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor, AutoProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        
        # 如果使用 LLaVA，使用 LLaVA 的 processor；否则使用原来的 vision_model processor
        if hasattr(args, 'llava_model') and args.llava_model is not None:
            print(f'Using LLaVA processor from {args.llava_model}')
            self.processor = AutoProcessor.from_pretrained(args.llava_model , use_fast=True)
            self.use_llava = True
            # 只考虑左填充：统一设置 tokenizer 的 padding_side 为 "left"
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
            # LLaVA prompt（与模型中的保持一致）
            self.llava_prompt = getattr(args, 'llava_prompt', 'Generate a comprehensive and detailed diagnosis report for this chest xray image.')
            self.prompt_length = 31  # 文本 prompt 的 token 数（不含 GT 报告）

            # 报告起始的特殊标记（放在报告最开头，用于在模型里快速定位 GT 起点）
            self.report_start_token = "<REPORT_START>"
            try:
                rs_ids = self.processor.tokenizer(
                    self.report_start_token,
                    add_special_tokens=False,
                )["input_ids"]
                # 统一成一维 list[int]
                if isinstance(rs_ids[0], list):
                    rs_ids = rs_ids[0]
                self.report_start_ids = rs_ids
                print(f"Report start token '{self.report_start_token}' ids: {self.report_start_ids}")
            except Exception as e:
                print(f"Warning: failed to get ids for report_start_token: {e}")
                self.report_start_ids = None
        else:
            print(f'Using vision encoder processor from {args.vision_model}')
            self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    # def _parse_image(self, img):
    #     pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
    #     return pixel_values[0]

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
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
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        to_return = {'id': features['id']}
        to_return['count'] = features.get("count")
        # to_return['viewposition'] = features.get("viewposition", "")
        to_return['indication'] = features.get("indication", "")
        to_return['prereport'] = features.get("prereport", "")
        # 训练集优先使用带分割标注的报告；其它集使用原 report
        if getattr(self, 'split', 'train') == 'train':
            input_text = self.clean_report(features.get("report_seg", features.get("report", "")))
        else:
            input_text = self.clean_report(features.get("report", ""))
        to_return['input_text'] = input_text
        
        if self.use_llava:
            # 对于 LLaVA，使用 processor 同时处理图像和文本
            # 构建 prompt：LLaVA 格式通常是 "USER: <image>\n{prompt}\nASSISTANT:"
            prompt_text = f"USER: <image>\n{self.llava_prompt}\nASSISTANT:"
            
            # 加载图像（只使用第一张，正面）
            image = Image.open(os.path.join(self.args.base_dir, features['image_path'][0])).convert("RGB")
            
            # 获取 max_length（默认 100，可以从 args 中获取）
            max_length = getattr(self.args, 'max_length', 100)
            
            # 构建 prompt 文本
            prompt_text = f"USER: <image>\n{self.llava_prompt}\nASSISTANT:"
            
            # 根据 split 决定是否包含目标文本
            split = getattr(self, 'split', 'train')
            if split == 'train':
                # 训练时：在报告开头插入一个特殊标记，后面所有 token 都视为 GT
                end_sym = getattr(self.args, 'end_sym', '')
                # 模板: [PROMPT] <REPORT_START> [GT 报告] [end_sym]
                full_prompt = f"{prompt_text} {self.report_start_token} {input_text}{end_sym}"

                processor_outputs = self.processor(
                    images=image,
                    text=full_prompt,
                    return_tensors="pt",
                    max_length=706,
                    padding="max_length",   # 左填充
                    truncation=True,
                )

                # 在 padded input_ids 中查找 report_start_token 的位置
                # 一定要能找到，因为我们显式在 GT 开头插入了 report_start_token
                assert self.report_start_ids is not None, "report_start_ids 未初始化"
                input_ids_1d = processor_outputs.input_ids.squeeze(0).tolist()  # [L]
                pat = self.report_start_ids
                L = len(input_ids_1d)
                P = len(pat)
                gt_start = None
                for i in range(L - P + 1):
                    if input_ids_1d[i:i+P] == pat:
                        gt_start = i + P   # GT 起点 = 特殊标记子序列结束之后
                        break
                assert gt_start is not None, "未在 input_ids 中找到报告起始标记 <REPORT_START>"
            else:
                # 验证/测试时：只使用 prompt（不包含目标文本）
                text_for_processor = prompt_text
                processor_outputs = self.processor(
                    images=image,
                    text=text_for_processor,
                    return_tensors="pt",
                    max_length=606,
                    padding="max_length",
                    truncation=True
                )


            
            # 保存处理后的结果（已经 padding 到 max_length）
            to_return["pixel_values"] = processor_outputs.pixel_values.squeeze(0)  # [C, H, W]
            to_return["input_ids"] = processor_outputs.input_ids.squeeze(0)  # [max_length]
            to_return["attention_mask"] = processor_outputs.attention_mask.squeeze(0)  # [max_length]
            # 文本 prompt 的 token 数（用于验证/测试生成时截断）
            to_return["prompt_length"] = self.prompt_length
            # 训练阶段：额外传递 GT 报告在 input_ids 中的起始位置（左填充前提下）
            if split == 'train':
                to_return["gt_start"] = int(gt_start)
        else:
            # 使用原来的 vision encoder processor
            images = []
            for image_path in features['image_path']:
                image = Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB")
                inputs = self.vit_feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)
                images.append(inputs)
            to_return["image"] = images
            # 单图像时复制一份以保持batch维度一致
            if(features.get("count")==1):
                images.append(images[0])

        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split  # 保存 split 信息
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.parser.split = split  # 传递给 parser

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset
