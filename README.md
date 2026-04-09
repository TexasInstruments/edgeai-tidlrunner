# edgeai-tidlrunner

Easy to use commandline tool for "Bring Your Own Edge AI Models" (BYOM) - to work with TI Deep Learning (TIDL) for TI MPU Processors for Edge AI. 


## Notice
* More details about TIDL and other tools are explained in [Edge AI developer landing page for MPUs](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu) and [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) - highly recommend to read these.


## Introduction

There are two packages in this repository with different purposes.
1. [tidlrunner](tidlrunner): Tool for Model Compilation, Inference, Analysis or Benchmark. 
2. tidloptimizer: Experimental tool for advanced model optimization (This is an experimental feature and recommend users not to use it as of now). 


## Documentation

Please go through the documentation of [tidlrunner](tidlrunner) to get started with this repository.

The core documentation for TI Deep Learning is available in the [edgeai-tidl-tools repo under /docs](github.com/TexasInstruments/edgeai-tidl-tools/tree/master/docs). Please refer there for detailed guidance on model compilation and inference arguments, supported operators, versioning debugging tips, and more. 


## What is new?
* 2026-April-03: [Model Inspector](./tidlrunner/edgeai_tidlrunner/modelinspector/README.md) is an interactive HTML visualization tool for analyzing ONNX models compiled with TIDL. Model Inspector provides comprehensive insights into model structure, performance, accuracy, and hardware acceleration.
