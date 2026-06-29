from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None):
        noise_type = os.environ.get('NOISE_TYPE', 'Symmetric')
        noise_level = os.environ.get('NOISE_LEVEL', '0.5')
        self.root = os.path.join(root, 'Cars196', noise_type, f'{noise_level}_{noise_type}')
        self.raw_root = os.path.join(root, 'Cars196', 'cars196')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes:
                resolved_path = self._resolve_image_path(im_path)
                self.im_paths.append(resolved_path)
                self.ys.append(y)
                self.I += [index]
                self.image_ids.append(im_path)
                index += 1

    def _resolve_image_path(self, im_path):
        direct_path = os.path.join(self.root, im_path)
        if os.path.isfile(direct_path):
            return direct_path

        filename = os.path.basename(im_path)
        stem, ext = os.path.splitext(filename)
        if stem.isdigit():
            global_idx = int(stem)
            if global_idx <= 8144:
                mapped = os.path.join(self.raw_root, 'cars_train', 'cars_train', f'{global_idx:05d}{ext}')
            else:
                mapped_idx = global_idx - 8144
                mapped = os.path.join(self.raw_root, 'cars_test', 'cars_test', f'{mapped_idx:05d}{ext}')
            if os.path.isfile(mapped):
                return mapped

        return direct_path
