# Candle-Bert
> Candle-Bert is a Python library that leverages the power of Rust and Python for fast and efficient natural language processing using BERT models. Currently, the package is tailored for Linux systems running Python 3.9. Contributions to extend compatibility are welcome.

## Benchmark comparsion

- Model:  `all-MiniLM-L6-V2`
- Device: `RTX-3070Ti-laptop`

| Library Used         | Time Per Sentence (seconds) | Result Shape |
| -------------------- | --------------------------- | ------------ |
| rust-bert            | 0.001040                    | (100, 384)   |
| Sentence Transformer | 0.002114                    | (100, 384)   |

## Installation

### Using pip

Install Candle-Bert via pip with the following command:

```bash
pip install candle-bert
```

This method currently supports only Linux for Python 3.9. Please consider contributing a pull request to support other versions or platforms.

### Compiling Locally

To install locally, you must first install Rust. After setting up Rust, proceed with the following steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/linkedlist771/Candle-Bert.git
   ```

2. Install `maturin`, a build system for Rust extensions in Python:
   ```bash
   pip install maturin
   ```

3. Build the project:
   ```bash
   maturin build --release
   ```

4. After the build process, a wheel file (`candle_bert-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl`) will be generated. Install this file using pip, or for development purposes, use:
   ```bash
   maturin develop
   ```

   This command installs the library locally for development use.

## Usage

Candle-Bert currently supports two main methods. Below is an example of how to use the library in a Python environment:

```python
import candle_bert
from time import perf_counter
import numpy as np
import random

MODEL_DIR_PATH = "Bert Model path"
model = candle_bert.CandleBert(model_dir_path=MODEL_DIR_PATH, use_cuda=True, use_pth=True, approximate_gelu=False)
sentence = "This is a demo sentence This is a demo sentenceThis is a demo sentenceThis is a demo sentence"
sentences = [random.choice(sentence.split()) for _ in range(100)]
start = perf_counter()
res = model.forward(sentences)
end = perf_counter()
print("Time per sentence: ", (end - start) / len(sentences))
res_tensor = np.array(res)
print("Res shape: ", res_tensor.shape)
```

### Performance

Using an RTX 3070Ti laptop with the `all-MiniLM-L6-V2` model, the performance metrics are as follows:

- **Rust-Bert**:
  - Time per sentence: `0.001040 seconds`
  - Result shape: `(100, 384)`

- **Sentence Transformer**:
  - Time per sentence: `0.002114 seconds`
  - Result shape: `(100, 384)`

These results demonstrate the efficient processing capabilities of Candle-Bert, especially when leveraging CUDA-enabled devices.

## Contributing

As I use this for my own projects, I know this might not be the perfect approach
for all the projects out there. If you have any ideas, just
[open an issue][issues] and tell me what you think.

If you'd like to contribute, please fork the repository and make changes as
you'd like. Pull requests are warmly welcome.

If your vision of a perfect `README.md` differs greatly from mine, it might be
because your projects are vastly different. In this case, you can create a
new file `README-yourplatform.md` and create the perfect boilerplate for that.

E.g. if you have a perfect `README.md` for a Grunt project, just name it as
`README-grunt.md`.



Before your PR, please format the script:

```bash
cargo fmt
```

## TODO：

- [ ]  Add the `mkl` compilation acceleration for the `CPU` case.

## Related projects

- [Candle]([huggingface/candle: Minimalist ML framework for Rust (github.com)](https://github.com/huggingface/candle))
- [Pyo3]([PyO3/pyo3: Rust bindings for the Python interpreter (github.com)](https://github.com/PyO3/pyo3))

## Licensing

This project is licensed under an  Apache License license. 
