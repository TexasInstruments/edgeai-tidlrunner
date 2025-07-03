
### edgeai_tidlrunner.runner.run is the Pythonic API of runner

```
kwargs = {
    'session': {
        'model_path': ./data/examples/models/mobilenet_v2.onnx',
    }
    'dataloader': {
        'name': 'image_classification_dataloader'
        'path': ./data/datasets/vision/imagenetv2c/val',
     }
}

edgeai_tidlrunner.runner.run('compile_model', **kwargs)
```

See the Pythonic example in [example_runner_py.py](./examples/vision/scripts/example_runner_py.py) which is invoked via [example_runner_py.sh](./example_runner_py.sh)



