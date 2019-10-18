from torchvision.models import resnet50
# pip install thop
# Support custome layer flops count
from thop import clever_format
from thop import profile
import torch

# pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
from ptflops import get_model_complexity_info

def test_resnet50_ptflops():
   net = resnet50()
   flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
   print('Flops:  ' + flops)
   print('Params: ' + params)

def test_resnet50_thop():
    model = resnet50()
    input = torch.randn(1,3,224,224)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops: ", flops, "params: ", params)

if __name__ == '__main__':
    test_resnet50_thop()
    test_resnet50_ptflops()
