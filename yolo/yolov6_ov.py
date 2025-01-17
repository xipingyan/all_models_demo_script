import openvino as ov
import cv2
import numpy as np

model_path = './yolov6_customer_model/yolov6s_int8.xml'

img = cv2.imread("dss1.png")
img = cv2.resize(img, (1920, 1088))
img = np.asarray(img)
input_tensor = np.expand_dims(img, 0)

print("input_tensor.shape = ", input_tensor.shape)

core = ov.Core()
model = core.read_model(model_path)

ppp = ov.preprocess.PrePostProcessor(model)
ppp.input().tensor().set_element_type(ov.Type.f16).set_layout(ov.Layout('NHWC'))  # noqa: N400

# 2) Here we suppose model has 'NCHW' layout for input
ppp.input().model().set_layout(ov.Layout('NHWC'))

# 3) Set output tensor information:
# - precision of tensor is supposed to be 'f32'
ppp.output().tensor().set_element_type(ov.Type.f32)

# 4) Apply preprocessing modifing the original 'model'
model = ppp.build()

compiled_model = core.compile_model(model, "GPU")
results = compiled_model.infer_new_request({0: input_tensor})

print("results=", results)