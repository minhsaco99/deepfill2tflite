#!/bin/bash
cd conversion
python torch_to_onnx.py
python onnx_to_tf.py
python tf_to_tflite.py