import onnxruntime as ort
import numpy as np

import onnx
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
# Refer: http://onnx.ai/sklearn-onnx/auto_examples/plot_intermediate_outputs.html

# Note:
# ONNX indeces accpet negative value.
# https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND
# Index values are allowed to be negative, as per the usual convention for counting backwards from the end, but are expected in the valid range.

my_output_name="/backbone/blocks/b0_l0/res/res.0/Concat_1_output_0"

# Load src model.
onnx_model = onnx.load("openvino.onnx")
onnx.checker.check_model(onnx_model)
all_outputs=[]
for out in enumerate_model_node_outputs(onnx_model):
    print(out)
    all_outputs.append(out)

# Check if my_output_name is in all outputs nodes.
if my_output_name in all_outputs:
    print("", my_output_name, "is in all_outpus")
    num_onnx = select_model_inputs_outputs(onnx_model, my_output_name)
    save_onnx_model(num_onnx, "tmp.onnx")

# Load sub model.
new_onnx_fn="tmp.onnx"
ort_sess = ort.InferenceSession(new_onnx_fn)

input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name
print("input_name=", input_name)
print("output_name=", output_name)

x = np.arange(1*1*3*8*224*224, dtype=float).reshape(1,1,3,8,224,224)
x = np.array(x).astype("float32")

# Run sub model and get nodel "my_output_name" result.
outputs = ort_sess.run(None, {'data': x})
print(f"get final result")
# print(outputs)
print("outputs", outputs)
print("outputs", type(outputs))
print("outputs", np.asarray(outputs).shape)


