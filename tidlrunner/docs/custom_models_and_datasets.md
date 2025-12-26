## Custom datasets and models

### Custom models using runner
It is fairly easy to compile custom datasets using runner. The user can point data_path their own folder and model_path to their own model. data_name can point to the appropriate dataloader, if a built-in dataloader can load the dataset used for that model. There are some generic dataloaders provided, that can be used to load a variety of dataset formats. 

<hr>

#### DataLoaders
##### Built-in image dataloaders
* image_files_dataloader: Image dataset from a folder - if accuracy computation is not needed, then this can be used (this does not provide any ground truth - so no accuracy will be computed when using this).
* image_classification_dataloader: This can be used in two ways (1) Image classification dataset from a folder of folders - images of each category is in its own folder. (2) A flat image folder and a txt file listing the image names and the category ids.
* coco_detection_dataloader: For COCO dataset Object Detection
* coco_segmentation_dataloader: For COCO dataset Semantic Segmentation

##### Built-in image dataloaders With ModelMaker / ModelComposer dataset formats
These are dataloaders for the dataset format used in [Edge AI Studio](dev.ti.com/modelcomposer) or [edgeai-tensorlab/edgeai-modelmaker](https://github.com/TexasInstruments/edgeai-tensorlab)
* For details see [MoelMaker / ModelComposer datasets](./modelmaker_datasets.md)
* modelmaker_classification_dataloader: For Image Classification 
* modelmaker_detection_dataloader: For Object Detection
* modelmaker_segmentation_dataloader: For Semantic/Instance Segmentation

#### Coming soon: Generic dataloaders (In development)
* Generic dataloaders using binary files such as pickle or numpy files.
* More information [Generic dataloader](./generic_dataloader.md)

<hr>

#### Preprocess
#### Built-in Image Preprocess functions
* no_preprocess
* image_preprocess
* image_classification_preprocess
* object_detection_preprocess
* semantic_segmentation_preprocess

#### Coming soon: Pythonic Callable as Preprocess (In development)
* It is also possible to provide a function for preprocess - this function when called should construct an object that supports __call__ method that does the preprocessing.

<hr>

#### Postprocess
#### Built-in Image Postprocess functions
* no_postprocess
* object_detection_postprocess

#### Coming soon: Pythonic Callable as Postprocess (In development)
* It is also possible to provide a function for postprocess - this function when called should construct an object that supports __call__ method that does the postprocessing.

<hr>
