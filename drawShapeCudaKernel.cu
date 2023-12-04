#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


__global__ void drawRectHollow(uchar4* data, int step, int x1, int y1, int x2, int y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= 1920 || y >= 960 )
        return;

    step = step/4;

    if(((y == y1 || y == y2) && (x >= x1 && x <= x2)) || ((x == x1 || x == x2) && (y <= y2 && y >= y1)))
    {
      data[y * step + x].x = 0;
      data[y * step + x].y = 0;
      data[y * step + x].z = 0;
      data[y * step + x].w = 0;
    }
}

__global__ void drawRectFilled(uchar4* data, int step, int x1, int y1, int x2, int y2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= 1920 || y >= 960 )
        return;

    step = step/4;

    if(y >= y1 && y < y2 && x >= x1 && x <= x2)
    {
      data[y * step + x].x = 0;
      data[y * step + x].y = 0;
      data[y * step + x].z = 0;
      data[y * step + x].w = 0;
    }
}

int main() {
    cv::Mat img = cv::imread("image.jpg");
    if (img.empty()) {
        fprintf(stderr, "Error: Could not load image.\n");
        return -1;
    }

    cv::resize(img, img, cv::Size(1920, 960));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::cvtColor(img, img, cv::COLOR_RGB2RGBA);

    int x1 = 400;
    int y1 = 300;
    int x2 = 500;
    int y2 = 550;

    cv::cuda::GpuMat d_img;
    d_img.upload(img);

    uchar4 *d_data = d_img.ptr<uchar4>();
    // cudaMalloc((void**)&d_data, sizeof(uchar4) * d_img.cols * d_img.rows);

    cudaMemcpy(d_data, d_img.data, sizeof(uchar4) * d_img.cols * d_img.rows, cudaMemcpyDeviceToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((d_img.cols + block_size.x - 1) / block_size.x, (d_img.rows + block_size.y - 1) / block_size.y);
    drawRectFilled<<<grid_size, block_size>>>(d_data, d_img.step, x1, y1, x2, y2);

    cudaMemcpy(d_img.data, d_data, sizeof(uchar4) * d_img.cols * d_img.rows, cudaMemcpyDeviceToDevice);

    d_img.download(img);
    cv::cvtColor(img, img, cv::COLOR_RGBA2RGB);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    cv::imshow("img", img);
    cv::waitKey(0);

    cudaFree(d_data);

    return 0;
}
