import os
from typing import Callable

import torch
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from feature_splatting.feature_extractor import MaskCLIPFeaturizer, batch_iterator


class FeatureExtractor:
    def __init__(
        self,
        clip_model_name: str,
        preprocess: Callable = None
    ):
        self.mask_clip = MaskCLIPFeaturizer(clip_model_name)
        self.sam_size = self.mask_clip.model.visual.conv1.out_channels
        self.mask_clip.model.eval()
        
        self.preprocess = preprocess if preprocess is not None else self.preprocess_img
    
    def preprocess_img(self, image, transform=None):
        if transform is None:
            norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transform = T.Compose([
                T.ToTensor(),
                norm
            ])
        
        if max(image.shape[:2]) > self.sam_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(self.sam_size * image.shape[1] / image.shape[0]), self.sam_size))
            else:
                image = cv2.resize(image, (self.sam_size, int(self.sam_size * image.shape[0] / image.shape[1])))
        image = image[:, :, ::-1]  # BGR to RGB

        raw_input_image = transform(Image.fromarray(image)) #
        return raw_input_image
    
    def __call__(self, path: str):
        if os.path.isfile(path):
            raise Exception("Single image embeddings are not supported yet!")
        elif os.path.isdir(path):
            images_paths = list(
                map(
                    lambda filename: os.path.join(path, filename),
                    filter(
                        lambda filename: any(filename.endswith(ext) for ext in [".jpeg", ".jpg", ".png"]),
                        os.listdir(path)
                    )
                )
            )
            return self.embed_set(images_paths)
        else:
            raise Exception(f"{path} does not exist or is neither a file nor a directory")
        
    """def embed_set(self, paths: list):
        embeddings = []
        with torch.no_grad():
            for image_path in paths:
                image = cv2.imread(image_path)
                raw_input_image = self.preprocess(image)
                #TODO: make batch processing
                embedding = self.mask_clip.model.encode_image(raw_input_image[None].cuda())[0].cpu()
                embeddings.append(embedding)
        return F.normalize(torch.stack(embeddings).mean(dim=0).float(), p=2, dim=0)"""
    
    def embed_set(self, paths: list, batch_size: int = 32):
        embeddings = []
        images = []
        
        # First load all images
        for image_path in paths:
            image = cv2.imread(image_path)
            raw_input_image = self.preprocess(image)
            images.append(raw_input_image)
        
        # Convert to tensor
        images = torch.stack(images)
        
        # Process in batches
        with torch.no_grad():
            for batch_imgs, in batch_iterator(batch_size, images):
                # Move batch to GPU and get embeddings
                batch_embeddings = self.mask_clip.model.encode_image(batch_imgs.cuda())
                embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all embeddings and compute mean
        embeddings = torch.cat(embeddings, dim=0)
        return F.normalize(embeddings.mean(dim=0).float(), p=2, dim=0)
