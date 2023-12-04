# Cuda Kernel Draw Shape
Draw a filled rectangle or a hollow rectangle (bounding box) usisng cuda programming.


## System Environment
- Hardware : Jetson Orin AGX
- OS : Ubuntu 20.04
- JetPack - 5.1.1
  - TensorRT - 8.5.2.2
  - Deepstream - 6.2
  - OpenCV - 4.6 + CUDA

## Installation
```
git clone https://github.com/RajUpadhyay/Basic-Cuda-Programming.git
cd Basic-Cuda-Programming
```

## compile
```
nvcc -o main -arch compute_87 drawShapeCudaKernel.cu `pkg-config --cflags --libs opencv4`
```

- To check your arch compute, run the following on the terminal
  - `nvcc --list-gpu-arch`