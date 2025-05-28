#include <iostream>
#include <cstdlib>

//image read/write libraries
extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image.h"
    #include "stb_image_write.h"
}

__global__ void grayscale_d(uint8_t *data, int width, int height, int channels) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);

    //because me might requisition threads we don't even need that are out of bounds
    if (x >= width || y >= height) return;

    //current index
    int i = (y*width + x)*channels;

    //0-255 value for gray
    uint8_t gray = static_cast<uint8_t>(
        0.299*data[i] + 
        0.587*data[i+1] + 
        0.114*data[i+2]
    );

    //apply to all color channels
    for (int j=0; j < channels; j++) {
            data[i + j] = gray;
    }
}
__host__ void grayscale_h(uint8_t *h_data, int width, int height, int channels) {
    size_t data_size = width * height * channels * sizeof(uint8_t);

    //copy host memory to device memory
    uint8_t *d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    //decode in parallel
    dim3 blockSize(16, 16); //blockSize.x, blockSize.y
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);//launch enough threads to cover each pixel
    grayscale_d<<<gridSize, blockSize>>>(d_data, width, height, channels);
    cudaDeviceSynchronize(); //halt until all cuda cores are done

    //copy host memory to device memory
    cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

__global__ void sobel_operator_d(uint8_t *in_data, uint8_t *out_data, int width, int height, int channels) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);

    //because me might requisition threads we don't even need that are out of bounds, and we don't want the very edges
    if (x >= width-1 || y >= height-1 || x <= 0 || y <= 0) return;
    
    //converts thread (x, y) to correct index in the flattened 2D array
    #define pix(x_offset, y_offset) ((y+y_offset) * width + (x+x_offset)) * channels
    //center is current pixel index and else is surround pixel indices
    int i[3][3] = { 
        {pix(-1, -1), pix(0, -1), pix(1, -1)},
        {pix(-1, 0), pix(0, 0), pix(1, 0)},
        {pix(-1, 1), pix(0, 1), pix(1, 1)}
    };

    //x and y filters
    int sobel_x = 
        in_data[i[0][0]]*-1 + in_data[i[0][2]]*1 +
        in_data[i[1][0]]*-2 + in_data[i[1][2]]*2 +
        in_data[i[2][0]]*-1 + in_data[i[2][2]]*1;
    int sobel_y = 
        in_data[i[0][0]]*-1 + in_data[i[0][1]]*-2 + in_data[i[0][2]]*-1 +
        in_data[i[2][0]]*1 + in_data[i[2][1]]*2 + in_data[i[2][2]]*1;
    //combine x and y
    int sobel_combined = abs(sobel_x*sobel_x) + abs(sobel_y*sobel_y);
    sobel_combined /= 12 ; //scale to 255

    //write to output
    for (int j=0; j < channels; j++) {
            out_data[i[1][1] + j] = (uint8_t) sobel_combined;
    }
}
__host__ void sobel_operator_h(uint8_t *h_data, int width, int height, int channels) {
    size_t data_size = width * height * channels * sizeof(uint8_t);
    
    //input array (copy host memory to device meory)
    uint8_t* d_data_in;
    cudaMalloc(&d_data_in, data_size);
    cudaMemcpy(d_data_in, h_data, data_size, cudaMemcpyHostToDevice);

    //empty output array
    uint8_t *d_data_out;
    cudaMalloc(&d_data_out, data_size);
    cudaMemcpy(d_data_out, h_data, data_size, cudaMemcpyHostToDevice);

    //decode in parallel
    dim3 blockSize(16, 16); //blockSize.x, blockSize.y
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);//launch enough threads to cover each pixel
    sobel_operator_d<<<gridSize, blockSize>>>(d_data_in, d_data_out, width, height, channels);
    cudaDeviceSynchronize(); //halt until all cuda cores are done

    //copy host memory to device memory
    cudaMemcpy(h_data, d_data_out, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data_in);
    cudaFree(d_data_out);
}

int main() {
    //name of file we're finding edges of
    char *file_name = (char*) "./images/doggy.jpg";

    //unpack image
    int width, height, channels;
    stbi_info(file_name, &width, &height, &channels);
    uint8_t *pixels = stbi_load(file_name, &width, &height, &channels, 0);

    //transform image
    printf("%d %d %d\n", width, height, channels);
    grayscale_h(pixels, width, height, channels);
    sobel_operator_h(pixels, width, height, channels);

    //write our output image
    stbi_write_png("output.png", width, height, channels, pixels, width * channels);
}
