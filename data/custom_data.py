import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, CLIPConfig
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image


# Define the dataset and data loader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class AIHub_data(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json,  transform=None, tokenizer=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        
        if self.transform is not None:
            # image = self.transform(image)
            image = self.transform(image=np.asarray(image))
            image = np.transpose(image['image'],(2, 0, 1))
        

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        del inputs['token_type_ids']
        for k in ['input_ids', 'attention_mask']:
            inputs[k] = torch.squeeze(inputs[k],0)
        
        inputs['pixel_values'] = image
        return inputs
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def __len__(self):
        return len(self.ids)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
                # A.Transpose((2, 0, 1))
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
                # A.Transpose((2, 0, 1))
            ]
        )
        
def build_loaders(mode, tokenizer):
    transforms = get_transforms(mode=mode)
    dataset = AIHub_data(
        root='/data/aihub/Training/images/',
        json='/data/aihub/Training/labels/labels.json',
        transform=transforms,
        tokenizer=tokenizer)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,
        shuffle=True if mode == "train" else False,
        # collate_fn=collate_fn
    )
    return dataloader