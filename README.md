# ONNX Operations Counter [WIP]

[![GitHub License](https://img.shields.io/badge/Apache-2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/onnx_opcounter)](https://pepy.tech/project/onnx_opcounter)
![PyPI](https://img.shields.io/pypi/v/onnx_opcounter.svg)

Counts number of parameters / MACs for ONNX models. 

## Installation

```bash
pip install onnx_opcounter
```

## Basic Usage

### Using CLI (calculate number of parameters)
```bash
onnx_opcounter {path_to_onnx_model}
```

### Using CLI (calculate number of parameters and MACs)
```bash
onnx_opcounter --calculate-macs {path_to_onnx_model}
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
