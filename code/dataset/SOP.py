from .base import *
import os

class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None):
        noise_type = os.environ.get('NOISE_TYPE', 'Symmetric')
        noise_level = os.environ.get('NOISE_LEVEL', '0.1')
        self.root = os.path.join(root, 'SOP', noise_type, f'{noise_level}_{noise_type}')
        self.raw_image_root = os.path.join(root, 'SOP', 'sop', 'Stanford_Online_Products')
        self.mode = mode
        self.transform = transform
        
        # Train: 0 - 11317 
        # Eval: 11318 - 22633 
        if self.mode == 'train':
            self.classes = range(0, 11318)
        elif self.mode == 'eval':
            self.classes = range(11318, 22634)  

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        
        txt_filename = 'Ebay_train.txt' if self.mode == 'train' else 'Ebay_test.txt'
        metadata_path = os.path.join(self.root, txt_filename)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            next(f)
            
            for line in f:
                parts = line.strip().split()
                image_id = int(parts[0])
                class_id = int(parts[1])
                path = parts[3]
                

                label = class_id - 1
                
                if label in self.classes:
                    self.ys += [label]
                    self.I += [image_id - 1]
                    
                    self.image_ids.append(str(image_id))

                    rel = path.replace('\\', '/')
                    under_noise = os.path.join(self.root, *rel.split('/'))
                    if os.path.isfile(under_noise):
                        full_path = under_noise
                    else:
                        full_path = os.path.join(self.raw_image_root, *rel.split('/'))
                    self.im_paths.append(full_path)

    def nb_classes(self):
        return len(self.classes)