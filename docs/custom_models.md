## Custom datasets and models

### Custom models using runner
It is fairly easy to compile custom datasets using runner. The user can point data_path their own folder and model_path to their own model. data_name can point to the appropriate dataloader, if a built-in dataloader can load the dataset used for that model. There are sone generic dataloaders provided, that can be used to load a variety of dataset formats. 

<hr>

#### DataLoaders
##### Built-in Image dataloaders
* image_files_dataloader: Generic Image dataset from a folder - if accuracy computation is not needed, then this can be used (this does not provide any ground truth - so not accuracy will be computed when using this).
* image_classification_dataloader: This can be used in two ways (1) Image classification dataset from a folder of folders - images of each category is in it's own folder. (2) A flat image folder and a txt file listing the image names and the category ids.
* modelmaker_classification_dataloader: For Generic Image Classification dataset format - for details see [Custom datasets](./custom_datasets.md)
* coco_detection_dataloader: For COCO dataset Object Detection
* modelmaker_detection_dataloader: For Generic Object Detection dataset format - for details see [Custom datasets](./custom_datasets.md)
* coco_segmentation_dataloader: For COCO dataset Semantic Segmentation
* modelmaker_segmentation_dataloader: For Generic Semantic/Instance Segmentation dataset format - for details see [Custom datasets](./custom_datasets.md)

#### Generic Callable as DataLoader 
It is also possible to provide a function for dataloader - this function when called should construct an object that supports __getitem__ and __len__ methods.

<hr>

#### Preprocess
#### Built-in Image Preprocess functions
* no_preprocess
* image_preprocess
* image_classification_preprocess
* object_detection_preprocess
* semantic_segmentation_preprocess

#### Generic Callable as Preprocess 
It is also possible to provide a function for preprocess - this function when called should construct an object that supports __call__ method that does the preprocessing.

<hr>

#### Postprocess
#### Built-in Image Postprocess functions
* no_postprocess
* object_detection_postprocess

#### Generic Callable as Preprocess 
It is also possible to provide a function for postprocess - this function when called should construct an object that supports __call__ method that does the preprocessing.

<hr>

### Custom models using rtwrapper
rtwrapper provides a fully flexible and low level access the runtimes. For maximum customization, this may be useful.
[rtwrapper advanced interface](./rtwrapper_interface.md)

<hr>
