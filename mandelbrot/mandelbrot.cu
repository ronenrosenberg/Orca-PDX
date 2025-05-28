#include <iostream>
#include <cstdlib>
#include <cuda/std/complex>

extern "C" {
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"

    #include "FPToolkit.c"
}

//globals (managed creates a global across both cpu and gpu)
__managed__ int WIDTH = 800;
__managed__ int HEIGHT = 800;
__managed__ int CHANNELS = 3;

__managed__ int MAX_ITER = 1000;
__managed__ double x_start = -2.0;
__managed__ double x_stop = 1.0;
__managed__ double y_start = -1.5;
__managed__ double y_stop = 1.5;


//the complex equation that gets drawn
__device__ cuda::std::complex<double> complex_eq(cuda::std::complex<double> z, cuda::std::complex<double> c) {
    return z*z+c;
    return z + log(pow(c, 314));
    return pow(z, 3) / (1.0 + cuda::std::norm(c*z)) + c;
    return z*z+atan(log(pow(c, 5)));
}

void scaler() {
    double scale = 0.01;
    double x_offset = 0.75;
    double y_offset = 0.665;
    x_start = x_start*scale + x_offset;
    x_stop = x_stop*scale + x_offset;
    y_start = y_start*scale + y_offset;
    y_stop = y_stop*scale + y_offset;
}

// void print_array(uint8_t *array, int len) {
//     for (int i=0; i<len; i++) {
//         printf("%d", array[i]);
//     }
// }

__device__ double remap(double x, double d_start, double d_stop, double r_start, double r_stop) {
    return (x-d_start) * (r_stop-r_start)/(d_stop-d_start) + r_start;
}

//writing pixels stuff
__device__ struct RGB {
    uint8_t r, g, b;

    // The constructor must also be __device__
    __device__ RGB(uint8_t red, uint8_t green, uint8_t blue) : r(red), g(green), b(blue) {}
};
__device__ void write_pixel(uint8_t *pixel, RGB rgb) {
    pixel[0] = rgb.r;
    pixel[1] = rgb.g;
    pixel[2] = rgb.b;
}

__global__ void fractal_d(uint8_t *pixels) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (x < WIDTH && y < HEIGHT) {
        //define z and c
        cuda::std::complex<double> c(remap(x, 0,WIDTH, x_start,x_stop), remap(y, 0,HEIGHT, y_start,y_stop));
        cuda::std::complex<double> z = 0;

        for (int i=0; i < MAX_ITER; i++) {
            if (z.real()*z.real() + z.imag()*z.imag() > 4) {
                double grayscale = abs((logf(i)/logf(MAX_ITER) * 255)-255);
                write_pixel(&pixels[(WIDTH*y + x) * CHANNELS], RGB(pow(grayscale,2), grayscale*.2, pow(grayscale,1.1)));
                return;
            }
            z = complex_eq(z, c);
        }
        write_pixel(&pixels[(WIDTH*y + x) * CHANNELS], RGB(0, 0, 0));
    }
}
void fractal_h(uint8_t *h_data) {
    size_t data_size = WIDTH * HEIGHT * CHANNELS * sizeof(uint8_t);

    //copy host memory to device memory
    uint8_t* d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    //decode in parallel
    dim3 blockSize(16, 16); //blockSize.x, blockSize.y
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);//launch enough threads to cover each pixel
    fractal_d<<<gridSize, blockSize>>>(d_data);
    cudaDeviceSynchronize(); //halt until all cuda cores are done

    //copy host memory to device memory
    cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}


int main() {
    uint8_t *pixels = new uint8_t[WIDTH*HEIGHT*CHANNELS];
    G_init_graphics(WIDTH, HEIGHT);

    while(1) {
        fractal_h(pixels);

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int i = (y * WIDTH + x) * 3;

                G_rgb(pixels[i]/255., pixels[i+1]/255., pixels[i+2]/255.);
                G_point(x, y);
            }
        }

        float x_zoom = (x_stop - x_start)/8;
        float y_zoom = (y_stop - y_start)/8;

        char key = G_wait_key();
        if (key == ',') {
            x_start += x_zoom;
            x_stop -= x_zoom;
            y_start += y_zoom;
            y_stop -= y_zoom;
        }
        else if (key == '.') {
            x_start -= x_zoom;
            x_stop += x_zoom;
            y_start -= y_zoom;
            y_stop += y_zoom;
        }
        else if (key == 's') {
            y_stop -= y_zoom;
            y_start -= y_zoom;
        }
        else if (key == 'w') {
            y_stop += y_zoom;
            y_start += y_zoom;
        }
        else if (key == 'a') {
            x_stop -= x_zoom;
            x_start -= x_zoom;
        }
        else if (key == 'd') {
            x_stop += x_zoom;
            x_start += x_zoom;
        }
        else if (key == 'q') {
            break;
        }

        G_display_image();
    }


    //write our output image
    //stbi_write_png("output.png", WIDTH, HEIGHT, CHANNELS, pixels, WIDTH*CHANNELS);
}
