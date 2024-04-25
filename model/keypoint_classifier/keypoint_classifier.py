#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf  # Add this line to import TensorFlow

class KeyPointClassifier(object):
    def __init__(self, model_path='FINALMODEL_cnn_classifier.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        # Ensure the input data matches the expected shape [1, 42]
        input_data = np.array([landmark_list], dtype=np.float32).reshape(1, 42)  # Reshape to [1, 42]
        
        # Set the tensor and invoke the interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get the output
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Convert logits to probabilities using softmax
        probabilities = tf.nn.softmax(result).numpy()
        result_index = np.argmax(probabilities[0])  # Get the index with the highest probability

        return result_index, probabilities[0, result_index]

    
