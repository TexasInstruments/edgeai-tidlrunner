## Custom dataset formats

The Custom/Generic dataset format is inspired by the [COCO dataset](https://cocodataset.org/) format. Images are to be present in the images folder and annotations are present in the json file.

This is also the same dataset format used by [Edge AI ModelComposer](https://dev.ti.com/modelcomposer/) and [edgeai-modelmaker](https://github.com/TexasInstruments/edgeai-tensorlab)

### Custom/Generic Image Classification dataset format
* modelmaker_classification_dataloader: This is the data_name (a.k.a dataloader.name) for Generic Image Classification dataset format.

The format is as follows:
 * images
   * images are in this folder
 * annotations
   * instances.json

Afield called category_id in each of the entries in the 'images' entry of the json file indicates the Ground Truth category for each image.

An example for this dataset is here: http://software-dl.ti.com/jacinto7/esd/modelzoo/11_00_00/datasets/vision/animal_classification.zip

### Custom/Generic Object Detection dataset format
* modelmaker_detection_dataloader: This is the data_name (a.k.a dataloader.name) for Generic Object Detection dataset format.

The format is as follows:
 * images
   * images are in this folder
 * annotations
   * instances.json

The 'bbox' fields in the json file indicates the bounding boxe for Object Detection Ground Truth.

An example for this dataset is here: http://software-dl.ti.com/jacinto7/esd/modelzoo/11_00_00/datasets/vision/tiscapes2017_driving.zip

### Custom/Generic Semantic Segmentation dataset format
* modelmaker_segmentation_dataloader: This is the data_name (a.k.a dataloader.name) for Generic Semantic/Instance Segmentation dataset format.

The format is as follows:
 * images
   * images are in this folder
 * annotations
   * instances.json

The 'segmentation' fields in the json file indicates the segmentation polygons for Semantic Segmentation or Instance Segmentation Ground Truth.

An example for this dataset is here: http://software-dl.ti.com/jacinto7/esd/modelzoo/11_00_00/datasets/vision/tiscapes2017_driving.zip

<hr>
