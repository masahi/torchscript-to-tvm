`test_bert.py` is a modification of the following notebook from onnxruntime repo:
https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/notebooks

run ```download_data.sh``` to get eval data, and you can run

`python test_bert.py`

TVM int8 eval result:
```
{'acc': 0.8529411764705882, 'f1': 0.8961937716262977, 'acc_and_f1': 0.8745674740484429}
```

PyTorch int8 eval result:
```
{'acc': 0.8578431372549019, 'f1': 0.8993055555555555, 'acc_and_f1': 0.8785743464052287}
```
