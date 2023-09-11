import numpy as np
import threading
import time
from queue import Queue
from PIL import Image, ImageDraw
import utils
from backends.backend import Backend


class TfliteBackend(Backend):
    def __init__(self, name, device="tpu"):
        super(TfliteBackend, self).__init__(name)
        self.precision = "int8"
        self.accelerator = "Edge TPU" if device=="tpu" else ""
        self.device = device
        if self.device == "tpu":
            from pycoral.adapters import common, classify
            from pycoral.utils.edgetpu import make_interpreter
            self.common = common
            self.classify = classify
            self.make_interpreter = make_interpreter
        else:
            import tflite_runtime
            import tflite_runtime.interpreter as tflite
            self.make_interpreter = tflite.Interpreter
    
    def get_accelerator(self):
        return self.accelerator

    def name(self):
        return self.name
    
    def version(self):
        import tflite_runtime
        return tflite_runtime.__version__
    
    def get_preprocess_func(self, model_name):
        model_names = ["resnet50", "mobilenet_v1", "mobilenet_v2", "mobilenet_v3", "inception_v1", "inception_v2", \
            "inception_v3", "inception_v4", "efficientnet_small_b0", "efficientnet_medium_b1", "efficientnet_large_b3"]
        if model_name not in model_names:
            raise ValueError(f"Please provide a valid model name from {model_names}")
        
        if model_name == "resnet50":
            return utils.preprocess_tflite_resnet
        else:
            return utils.preprocess_tflite_mobilenet

    def load_backend(self, model_path, model_name=None):
        self.model_name = model_name
        self.interpreter = self.make_interpreter(model_path)
        
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        params = self.input_details['quantization_parameters']
        self.input_scale = params['scales']
        self.input_zero_point = params['zero_points']
    
    def __call__(self, inputs):
        inputs = inputs / self.input_scale + self.input_zero_point
        inputs = inputs.astype(np.uint8)
        if self.device == "tpu":        
            self.common.set_input(self.interpreter, inputs)
        else:
            inputs = np.expand_dims(inputs, axis=0)
            self.interpreter.set_tensor(self.input_details['index'], inputs)
    
        start = time.time()
        self.interpreter.invoke()
        end = time.time()
        if self.device == "tpu":
            classes = self.classify.get_classes(self.interpreter, 1, 0.0)
        else:
            classes = self.interpreter.get_tensor(self.output_details['index'])
        return classes, end - start
    
    def warmup(self, inputs, warmup_steps=100):
        for step in range(warmup_steps):
            self(inputs)
    
    def capture_stats(self):
        self.stop_event = threading.Event()
        self.output_queue = Queue()
        if self.device=="tpu":
            self.psutil_thread = threading.Thread(
                target=utils.get_coral_stats, args=(self.output_queue, self.stop_event), daemon=True
            )
        else:
            self.psutil_thread = threading.Thread(
                target=utils.get_stats_rockpi, args=(self.output_queue, self.stop_event), daemon=True
            )
        self.psutil_thread.start()
    
    def get_avg_stats(self):
        ram_usage, cpu_util, temp, tpu_freq, cpu_freq = [], [], [], [], []
        
        while not self.output_queue.empty():
            if self.device == "tpu":
                c,r,t, tf, cf = self.output_queue.get()
            else:
                c,r,t, cf = self.output_queue.get()
            ram_usage.append(r)
            cpu_util.append(c)
            temp.append(t)
            if self.device == "tpu":
                tpu_freq.append(tf)
            cpu_freq.append(cf)
        ram_usage, cpu_util, temp = np.array(ram_usage), np.array(cpu_util), np.array(temp)
        
        if self.device == "cpu":
            tpu_freq = ""
        stats = {
            "cpu": cpu_util,
            "memory": ram_usage,
            "temperature": temp,
            "tpu_freq": tpu_freq,
            "cpu_freq": cpu_freq
        }
        return stats
    
    def get_pred(self, outputs):
        if self.device == "tpu":
            pred = outputs[0].id
        else:
            pred = np.argmax(np.array(outputs))
        if self.model_name == "resnet50":
            return pred
        return pred - 1
    
    def destroy(self):
        del self.interpreter



class TfliteDetectorBackend(TfliteBackend):
    def __init__(self, name, device="tpu"):
        super(TfliteBackend, self).__init__(name)
        self.precision = "int8"
        self.accelerator = "Edge TPU" if device=="tpu" else ""
        self.device = device
        if self.device == "tpu":
            from pycoral.adapters import common, detect
            from pycoral.utils.edgetpu import make_interpreter
            self.make_interpreter = make_interpreter
        else:
            import tflite_runtime
            import tflite_runtime.interpreter as tflite
            import helpers.tflite_common as common
            import helpers.tflite_detect as detect
            self.make_interpreter = tflite.Interpreter
        
        self.common = common
        self.detect = detect
    
    def get_accelerator(self):
        return self.accelerator

    def name(self):
        return self.name
    
    def version(self):
        import tflite_runtime
        return tflite_runtime.__version__
    
    def get_preprocess_func(self, model_name):
        model_names = ["resnet50", "mobilenet_v1", "mobilenet_v2", "mobilenet_v3", "inception_v1", "inception_v2", \
            "inception_v3", "inception_v4", "efficientnet_small_b0", "efficientnet_medium_b1", "efficientnet_large_b3"]
        if model_name not in model_names:
            raise ValueError(f"Please provide a valid model name from {model_names}")
        
        if model_name == "resnet50":
            return utils.preprocess_tflite_resnet
        else:
            return utils.preprocess_tflite_mobilenet

    def load_backend(self, model_path, model_name=None):
        self.model_name = model_name
        self.interpreter = self.make_interpreter(model_path)
        
        self.interpreter.allocate_tensors()

    
    def __call__(self, inputs, output="out.png"):
        print(inputs.size)
        _, scale = self.common.set_resized_input(
            self.interpreter, inputs.size, lambda size: inputs.resize(size, Image.LANCZOS))
        
        start = time.time()
        self.interpreter.invoke()
        end = time.time()

        objs = self.detect.get_objects(self.interpreter, 0.6, scale)
        if not objs:
            print('No objects detected')

        for obj in objs:
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)

        if output:
            image = inputs.convert('RGB')
            self.draw_objects(ImageDraw.Draw(image), objs)
            image.save(output)
        return objs, end - start
    
    def draw_objects(self, draw, objs):
        """Draws the bounding box and label for each object."""
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                        outline='red')
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                    '%s\n%.2f' % (obj.id, obj.score),
                    fill='red')
    
    