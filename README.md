# ONNX Operations Counter

Counts number of parameters / FLOPS for ONNX models. 

[WIP]

## Installation

```bash
pip install onnx_opcounter
```

## Basic Usage

### Using CLI (calculate number of parameters)
```bash
onnx_opcounter {path_to_onnx_model}
```

### Using CLI (calculate number of parameters and FLOPS)
```bash
onnx_opcounter --calculate-flops {path_to_onnx_model}
```

### Using API
```python
from onnx_opcounter import calculate_params
import onnx

model = onnx.load_model('./path/to/onnx/model')
params = calculate_params(model)

print('Number of params:', params)
```

## License
The software is covered by Apache License 2.0.
