# ONNX Operations Counter

Counts number of parameters / FLOPS for ONNX models.

## Installation

```bash
pip install onnx_opcounter
```

## Basic Usage

### Using CLI
```bash
onnx_opcounter parameters {path_to_onnx_model}
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
