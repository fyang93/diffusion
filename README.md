This is a faster and improved version of diffusion retrieval, inspired by [diffusion-retrieval](https://github.com/ahmetius/diffusion-retrieval).

Reference:
- [F. Yang](https://fyang.me/about), [R. Hinami](http://www.satoh-lab.nii.ac.jp/member/hinami/), [Y. Matsui](http://yusukematsui.me), S. Ly, [S. Satoh](http://research.nii.ac.jp/~satoh/index.html), "**Efficient Image Retrieval via Decoupling Diffusion into Online and Offline Processing**", AAAI 2019.

## Features

- All random walk processes are moved to offline, making the online search remarkably fast

- In contrast to previous works, we achieved better performance by applying late truncation instead of early truncation to the graph

## Requirements

- Install Facebook [FAISS](https://github.com/facebookresearch/faiss) by running `conda install faiss-cpu -c pytorch`
> Optional: install the faiss-gpu under the [instruction](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) according to your CUDA version

- Install joblib by running `conda install joblib`

- Install tqdm by running `conda install tqdm`

## Parameters

All parameters can be modified in `Makefile`. You may want to edit `DATASET` and `FEATURE_TYPE` to test all combinations of each dataset and each feature type.
Another parameter `truncation_size` is set to 1000 by default, for large datasets like Oxford105k and Paris106k, change it to 5000 will improve the performance.

## Run

- Run `make download` to download files needed in experiments;

- Run `make mat2npy` to convert .mat files to .npy files;

- Run `make rank` to get the results. If you have GPUs, try using commands like `CUDA_VISIBLE_DEVICES=0,1 make rank`, `0,1` are examples of GPU ids.
> Note: on Oxford5k and Paris6k datasets, the `truncation_size` parameter should be no larger than 1024 when using GPUs according to FAISS's limitation. You can use CPUs instead.

## Authors

- [Fan Yang](https://fyang.me/about) wrote the algorithm
- [Ryota Hinami](http://www.satoh-lab.nii.ac.jp/member/hinami/) wrote the evaluator

