from mnist_net import *
import onnx
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = '/data/home/mjnn/majianan/ProvRepair/model/mnist/mnist_relu_6_100.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# # Or you can load a regular onnx model and pass it to the converter
# onnx_model = onnx.load(onnx_model_path)
# torch_model_2 = convert(onnx_model)

print(torch_model_1)
check = []
for name, para in torch_model_1.named_parameters():
    print(name, para.shape)
    check.append(para)

model = FNN_6_100()
ind = 0
real_check = {}
for name, para in model.named_parameters():
    if check[ind].shape == para.shape:
        real_check[name] = check[ind]
    else:
        print('error')
        print(check[ind].shape, para.shape)
    ind += 1
model.load_state_dict(real_check)
print(model.state_dict())
torch.save(model.state_dict(), '/data/home/mjnn/majianan/ProvRepair/model/mnist/mnist_relu_6_100.pth')