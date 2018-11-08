This is a faster and improved version of diffusion retrieval, inspired by [diffusion-retrieval](https://github.com/ahmetius/diffusion-retrieval).

Reference:
- [F. Yang](https://fyang.me/about), [R. Hinami](http://www.satoh-lab.nii.ac.jp/member/hinami/), [Y. Matsui](http://yusukematsui.me), S. Ly, [S. Satoh](http://research.nii.ac.jp/~satoh/index.html), "**Efficient Image Retrieval via Decoupling Diffusion into Online and Offline Processing**", AAAI 2019.

## Features

- All random walk processes are moved to offline, making the online search remarkably fast

- In contrast to previous works, we achieved better performance by applying late truncation instead of early truncation to the graph

## Requirements

Facebook [FAISS](https://github.com/facebookresearch/faiss) is used to build kNN graph.
Install faiss-cpu or faiss-gpu according to your situation.
You may also need joblib to run multiple processes in parallel.
You can install joblib with `pip` or `conda`.

## Run

- Run `make download` to download files needed in experiments;

- Run `make mat2npy` to convert .mat files to .npy files;

- Run `make rank` to get the results. If you have GPUs, try using commands like `CUDA_VISIBLE_DEVICES=0,1 make rank`, `0,1` are examples of GPU ids.
> Note: on Oxford5k and Paris6k datasets, the `truncation_size` parameter should be no larger than 1024 when using GPUs according to FAISS's limitation. You can use CPUs instead.

## Author

- [Fan Yang](https://fyang.me/about)

