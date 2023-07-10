#-----------------------------------------------------------------------------
# Dataset and Dataloader related function and class
#-----------------------------------------------------------------------------

import torch, torchvision
from collections import Counter
from random import seed, choice, sample
import json, PIL, sys
from random import seed, choice, sample

#-----------------------------------------------------------------------------
# vocabulary
#-----------------------------------------------------------------------------
class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.max_word_length = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.max_word_length = max(self.max_word_length, len(word))

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def make(self, dataset, min_word_freq, karpathy_json_path="./Karpathy_splits", max_len=100):
        r"""
        Creates input files for training, validation, and test data.
        :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
        :param karpathy_json_path: path of Karpathy JSON file with splits and captions
        :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
        :param max_len: don't sample captions longer than this length
        """
        assert dataset in ('coco', 'flickr8k', 'flickr30k')

        # Read Karpathy JSON
        with open(f"{karpathy_json_path}/dataset_{dataset}.json", 'r') as j:
            data = json.load(j)

        # Read image paths and captions for each image
        train_image_paths = []
        train_image_captions = []
        val_image_paths = []
        val_image_captions = []
        test_image_paths = []
        test_image_captions = []
        word_freq = Counter()

        for img in data['images']:
            for c in img['sentences']:
                # Update word frequency
                word_freq.update(c['tokens'])

        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        self.add_word("<pad>")
        for word in words:
            self.add_word(word)
        self.add_word("<unk>")
        self.add_word("<start>")
        self.add_word("<end>")

#-----------------------------------------------------------------------------
# dataset and dataloader
#-----------------------------------------------------------------------------

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder, dataset, split, vocab, isLimit, transform=None, device=None):
        r"""
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder = data_folder
        assert dataset in ('coco', 'flickr8k', 'flickr30k')
        self.dataset = dataset
        self.split = split
        assert self.split in ("train", "val", "test"), f"bad split key word: split={split}"
        self.vocab = vocab
        self.isLimit = isLimit
        self.transform = transform
        self.device = device

        self.prepare()
        #print(self.images.shape, self.captions.shape, self.lengths.shape)

    def __len__(self):
        return self.captions.size(0)

    def __getitem__(self, i):

        img = self.images[i//self.captions_per_image,:,:,:]
        caption = self.captions[i,:]
        capleng = self.lengths[i]

        if self.split is "train":
            return img, caption, capleng
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            bias = (i // self.captions_per_image) * self.captions_per_image
            all_captions = self.captions[bias:(bias+self.captions_per_image)]
            return img, caption, capleng, all_captions


    def prepare(self, captions_per_image=5, min_word_freq=5, max_len=98, karpathy_json_path="./Karpathy_splits/"):

        self.captions_per_image = captions_per_image
        data_folder  = self.data_folder
        dataset  = self.dataset
        vocab = self.vocab
        image_folder = f"{data_folder}/images/"

        #-- Read Karpathy JSON
        with open(f"{karpathy_json_path}/dataset_{dataset}.json", 'r') as j:
            data = json.load(j)

        image_paths = []
        image_captions = []

        for img in data["images"]:
            if img["split"] not in (self.split,):
                continue

            captions = []
            for c in img["sentences"]:
                if len(c["tokens"]) <= max_len:
                    captions.append(c["tokens"])
            if len(captions) == 0:
                continue

            path = f"./{data_folder}/{dataset}/images/{img['filename']}"
            image_paths.append(path)
            image_captions.append(captions)

        # Sanity check
        assert len(image_paths) == len(image_captions), "bad lengths."

        #-- limit
        if self.isLimit:
            limit = 100 if len(image_paths) > 2000 else 20
            image_paths = image_paths[:limit]
            image_captions = image_captions[:limit]

        #-- prepare images
        images = []
        for dir in image_paths:
            img = PIL.Image.open(dir)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img.unsqueeze(0))
        images = torch.cat(images, dim=0).type( torch.FloatTensor )

        #-- prepare captions, caption lengths
        encoded_captions = []
        caption_lengths = []
        for caps in image_captions:
            caps = list(caps)
            # Sample captions
            if len(caps) < captions_per_image:
                captions = caps + [choice(caps) for _ in range(captions_per_image - len(caps))]
            else:
                captions = sample(caps, k=captions_per_image)
            # Sanity check
            assert len(captions) == captions_per_image

            for j, c in enumerate(captions):
                enc_c = [vocab('<start>')] + [vocab(word) for word in c] + [vocab('<end>')] + [vocab('<pad>')] * (max_len - len(c))
                # Find caption lengths, <start> ... <end>
                c_len = len(c) + 2

                encoded_captions.append( enc_c )
                caption_lengths.append( c_len )
        encoded_captions = torch.tensor( encoded_captions ).type( torch.LongTensor )
        caption_lengths = torch.tensor( caption_lengths ).type( torch.LongTensor )

        #-- to device
        if self.device is not None:
            images = images.to(self.device)
            encoded_captions = encoded_captions.to(self.device)
            caption_lengths = caption_lengths.to(self.device)

        #-- set attibutes
        self.files = image_paths
        self.image_captions = image_captions
        self.images = images
        self.captions = encoded_captions
        self.lengths = caption_lengths

        print(f"Loaded {self.images.size(0)} images and {self.captions.size(0)} captions for {self.split}.")






def get_dataloaders(Ps, vocab):
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((256,256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],  std =[0.229, 0.224, 0.225]),
                    ])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(Ps["data_folder"], dataset=Ps["dataset"], split="train", vocab=vocab, isLimit=Ps["isLimit"], transform=transform, device=Ps["device"]),
        batch_size=Ps["batch_size"], shuffle=True, num_workers=Ps["num_workers"], drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        CaptionDataset(Ps["data_folder"], dataset=Ps["dataset"], split="val", vocab=vocab, isLimit=Ps["isLimit"], transform=transform, device=Ps["device"]),
        batch_size=Ps["batch_size"], shuffle=True, num_workers=Ps["num_workers"], drop_last=True)

    return {"train" : train_loader, "valid": valid_loader}

#-----------------------------------------------------------------------------
# debug
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    ps = {
        "data_folder": '/data/Flickr8k/',
        "dataset": 'flickr8k',
        "isLimit": False,
        "batch_size":16,
        "device":'cuda'
    }
    data_loader = get_dataloaders(ps, Vocabulary())
    pass