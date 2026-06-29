import os
from .base import *

class CUBirds(BaseDataset):
    def __init__(self, root, mode, transform=None):
        noise_type = os.environ.get('NOISE_TYPE', 'Symmetric')
        noise_level = os.environ.get('NOISE_LEVEL', '0.9')
        self.root = os.path.join(root, 'CUB', noise_type, f'{noise_level}_{noise_type}')
        self.raw_image_root = os.path.join(root, 'CUB', 'cub', 'CUB_200_2011', 'images')
        self.mode = mode
        self.transform = transform
        
        if self.mode == 'train':
            self.classes = set(range(0, 100))
        elif self.mode == 'eval':
            self.classes = set(range(100, 200))
        
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        
        # 1. Read images.txt
        images_txt_path = os.path.join(self.root, 'images.txt')
        image_id_to_path = {}
        with open(images_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    path = parts[1]
                    path = path.replace('\\', '/')
                    image_id_to_path[image_id] = path

        # 2. Read image_class_labels.txt (Noisy labels used for training)
        labels_txt_path = os.path.join(self.root, 'image_class_labels.txt')
        image_id_to_label = {}
        with open(labels_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    label = int(parts[1]) - 1 
                    image_id_to_label[image_id] = label

        # 3. Integrate
        index = 0
        for image_id in image_id_to_path:
            if image_id in image_id_to_label:
                label = image_id_to_label[image_id]
                path = image_id_to_path[image_id]
                
                if label in self.classes:
                    rel = path.replace('\\', '/')
                    under_noise = os.path.join(self.root, 'images', *rel.split('/'))
                    if os.path.isfile(under_noise):
                        full_path = under_noise
                    else:
                        full_path = os.path.join(self.raw_image_root, *rel.split('/'))
                    
                    self.ys.append(label)
                    self.I.append(index)
                    self.im_paths.append(full_path)
                    self.image_ids.append(image_id) # Store ID for verification
                    index += 1
                    
        print(f"[{self.mode}] Loaded {len(self.im_paths)} images from {len(self.classes)} classes.")