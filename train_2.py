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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define the dataset and data loader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CLIP model and processor
# model_name = "openai/clip-vit-base-patch32"
model_name = "EleutherAI/polyglot-ko-1.3b"
# processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# config = CLIPConfig.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)

# _model = CLIPModel.from_pretrained(model_name)
# print(_model.config.max_position_embeddings)
# config.max_position_embeddings = 128
# config.text_config.max_position_embeddings = 128

# model = CLIPModel(config)

# model.load_state_dict(_model.state_dict())
# model = model.to(device)
# print(model.config.max_position_embeddings)
# Freeze all layers except the projection layer
for name, param in model.named_parameters():
    if "visual" in name and "projection" not in name:
        param.requires_grad = False
    elif "text" in name and "projection" not in name:
        param.requires_grad = False

optimizer = AdamW(
    [{'params':model.visual_projection.parameters()},
     {'params':model.text_projection.parameters()}
     ],
    lr=0.0001
    )

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
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
train_loader = build_loaders('train', tokenizer)


# Training loop
model.train()
for epoch in range(10):
    total_loss = 0.0
    for batch, inputs in enumerate(train_loader):
        inputs = {k:v.to('cuda:1') for k, v in inputs.items()}
        # inputs['pixel_values'] = inputs['pixel_values'].to(device)
        

        optimizer.zero_grad()
        # Preprocess images
        
        # Forward 
        outputs = model(**inputs)
        
        batch_size = inputs['pixel_values'].size(0)
        logits_per_image = outputs.logits_per_image

        # Generate random labels for contrastive learning
        targets = torch.arange(batch_size).to(device)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits_per_image / 0.1, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        
        if (batch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{10}], Step [{batch+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{10}], Total Loss: {total_loss:.4f}")
    # break

# Save the trained model
# model.save_pretrained("path/to/save/model")
