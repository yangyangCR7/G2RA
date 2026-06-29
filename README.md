# G2RA

## Datasets

Download the following datasets:
 - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
 - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
 - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
If the official links above are broken, you can download all the required data from this mirror repository:([Link](https://huggingface.co/datasets/XiN0919/FGVC/tree/main)).


Place them in `Dataset` dir as follows:
```
Dataset/
├── Cars196/
│   ├── Symmetric/
│   │   ├── 0.1_Symmetric/
│   │   │   ├── car_ims/
│   │   │   └── cars_annos.mat
│   │   ├── ...
│   │   └── 0.9_Symmetric/
│   └── SmallCluster/
├── CUB/
│   ├── Symmetric/
│   │   ├── 0.1_Symmetric/
│   │   │   ├── images/
│   │   │   │   ├── 001.Black_footed_Albatross/
│   │   │   │   ├── ...
│   │   │   │   └── 200.Common_Yellowthroat/
│   │   │   ├── image_class_labels.txt
│   │   │   └── images.txt
│   │   ├── ...
│   │   └── 0.9_Symmetric/
│   └── SmallCluster/
└── SOP/
    ├── Symmetric/
    │   ├── 0.1_Symmetric/
    │   │   ├── bicycle_final/
    │   │   ├── ...
    │   │   ├── toaster_final/
    │   │   ├── Ebay_train.txt
    │   │   └── Ebay_test.txt
    │   ├── ...
    │   └── 0.9_Symmetric/
    └── SmallCluster/
```
