from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from dataset.data_helper import create_datasets
import torch


def custom_collate_fn(batch):
    """Handle variable-length items while keeping tensor fields stacked."""
    collated = {}

    # Keep variable-length fields as plain lists; everything else uses default_collate
    list_fields = ['masks', 'seg_organs', 'input_text', 'id', 'sam_original_size', 'sam_resize_size']

    for key in list_fields:
        if key in batch[0]:
            collated[key] = [sample[key] for sample in batch]

    # Collate the remaining (tensor/number) fields normally
    batch_for_collate = []
    for sample in batch:
        sample_copy = {k: v for k, v in sample.items() if k not in list_fields}
        batch_for_collate.append(sample_copy)

    if batch_for_collate and batch_for_collate[0]:
        collated.update(default_collate(batch_for_collate))

    return collated


def llava_collate_fn(batch):
    """Placeholder collate for LLaVA (default_collate is sufficient)."""
    return default_collate(batch)



class DataModule(LightningDataModule):

    def __init__(
            self,
            args,
    ):
        super().__init__()
        self.args = args

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, dev_dataset, test_dataset, sam_dataset = create_datasets(self.args)
        self.dataset = {
            "train": train_dataset, "validation": dev_dataset, "test": test_dataset
        }


    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        # 如果使用 LLaVA，使用自定义 collate_fn 处理不同长度的序列
        collate_fn = llava_collate_fn if (hasattr(self.args, 'llava_model') and self.args.llava_model is not None) else custom_collate_fn
        loader = DataLoader(
            self.dataset["train"], 
            batch_size=self.args.batch_size, 
            drop_last=True, 
            pin_memory=True,
            num_workers=self.args.num_workers, 
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=collate_fn
        )
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        collate_fn = llava_collate_fn if (hasattr(self.args, 'llava_model') and self.args.llava_model is not None) else custom_collate_fn
        loader = DataLoader(
            self.dataset["validation"], 
            batch_size=self.args.val_batch_size, 
            drop_last=False, 
            pin_memory=True,
            num_workers=self.args.num_workers, 
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=collate_fn
        )
        return loader


    def test_dataloader(self):
        collate_fn = llava_collate_fn if (hasattr(self.args, 'llava_model') and self.args.llava_model is not None) else custom_collate_fn
        loader = DataLoader(
            self.dataset["test"], 
            batch_size=self.args.test_batch_size, 
            drop_last=False, 
            pin_memory=False,
            num_workers=self.args.num_workers, 
            prefetch_factor=self.args.prefetch_factor,
            collate_fn=collate_fn
        )
        return loader
