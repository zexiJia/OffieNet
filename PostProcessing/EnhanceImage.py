import torch
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from os import path
from pycuda.driver import Stream

# magic_code = torch.cuda.FloatTensor(8)
pi = np.pi

code = '''
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define sigma 4.5
#define psi 0
#define gamma 0.5
#define pi 3.141592653589793

// 要修改 这三个值都要改
#define padding 12
#define ksize 25
#define shared_size 56

struct GaborKenel{
    float cos_kernel;
    float sin_kernel;
};

__device__ GaborKenel computeGaborKenel(float theta, int x,  int y, float Lambda=8){

    GaborKenel gk;
    float sigma_x = sigma;
    float sigma_y = sigma / gamma;
    float x_theta = x * cosf(theta) + y * sinf(theta);
    float y_theta = -x * sinf(theta) + y * cosf(theta);

    gk.cos_kernel = expf(-0.5 * ((x_theta*x_theta) / (sigma_x *sigma_x) + (y_theta *y_theta) / (sigma_y *sigma_y))) * cosf(2 * pi / Lambda * x_theta + psi);
    gk.sin_kernel = expf(-0.5 * ((x_theta*x_theta) / (sigma_x *sigma_x) + (y_theta *y_theta) / (sigma_y *sigma_y))) * sinf(2 * pi / Lambda * x_theta + psi);

    return gk;
}


__device__ float bilinearInterplation(float *src, int scale, int dstX, int dstY, int srcBs, int srcH, int srcW, int total_channel=1, int channel_index=0){

    float srcX = (dstX + 0.5) / scale - 0.5;
    float srcY = (dstY + 0.5) / scale - 0.5;

    int modXi = (int) floorf(srcX);
    int modYi = (int) floorf(srcY);

    modXi = max(modXi, 0);
    modYi = max(modYi, 0);

    float modXf = srcX - modXi;
    float modYf = srcY - modYi;
    int modXiPlusOneLim = min(modXi+1, srcH-1);
    int modYiPlusOneLim = min(modYi+1, srcW-1);
    
    int srcStartOffset = (srcH * srcW * channel_index);
    float tl = *(src + srcBs * (srcH * srcW * total_channel) + srcW * modXi + modYi + srcStartOffset);
    float tr = *(src + srcBs * (srcH * srcW * total_channel) + srcW * modXi + modYiPlusOneLim + srcStartOffset);
    float bl = *(src + srcBs * (srcH * srcW * total_channel) + srcW * modXiPlusOneLim + modYi + srcStartOffset);
    float br = *(src + srcBs * (srcH * srcW * total_channel) + srcW * modXiPlusOneLim + modYiPlusOneLim + srcStartOffset);

    float b = modYf * br + (1 - modYf) * bl;
    float t = modYf * tr + (1 - modYf) * tl;

    return  modXf* b + (1 - modXf) * t;
}


__device__ void convolution(float *data, float theta, float Lambda, float *rescos, float *ressin, float *resenh){

    float rcos = 0;
    float rsin = 0;
    for(int i = -padding ; i <= padding; i++ ){
        for(int j = -padding ; j <= padding; j++){
            GaborKenel gk = computeGaborKenel(theta, i, j, Lambda);            
            float d = *(data + (threadIdx.x + i + padding) * shared_size + threadIdx.y + j + padding);
            rcos += gk.cos_kernel * d;
            rsin += gk.sin_kernel * d;
        }
    }
    *resenh = rcos;
    *rescos = atan2f(rcos, rsin);
    *ressin = sqrtf(rsin * rsin + rcos * rcos);
}


__global__ void EnhanceImageProduce(float *images, float *ori, float *fre, float *enhIc, float *enhIs, float *enhI){

    // GridDim.z is batchsize and GridDim.x * blockDim.x = h, etc.
    // one thread resposes to a position

    int batch = blockIdx.z;
    int outx = blockIdx.x * blockDim.x + threadIdx.x;
    int outy = blockIdx.y * blockDim.y + threadIdx.y;
  
    int rw = gridDim.y * blockDim.y;
    int rh = gridDim.x * blockDim.x ;

    int h = rh + 2*padding;
    int w = rw + 2*padding;

    float *rescos = enhIc +  batch * (rw * rh) + outx * rw + outy; // double channels
    float *ressin = enhIs +  batch * (rw * rh) + outx * rw + outy; // double channels
    float *resenh = enhI +  batch * (rw * rh) + outx * rw + outy; // double channels

    __shared__ float data[shared_size * shared_size];
    for(int i = threadIdx.x; i < shared_size; i+=blockDim.x ){
        for(int j = threadIdx.y; j < shared_size; j+=blockDim.y){
            data[i*shared_size + j] = *(images + batch*h*w + (blockIdx.x*blockDim.x + i) * w + (blockIdx.y*blockDim.y + j));
        }
    }
    __syncthreads();

   float cos2theta = bilinearInterplation(ori, 8, outx, outy, batch, rh / 8, rw / 8, 2, 0);
   float sin2theta = bilinearInterplation(ori, 8, outx, outy, batch, rh / 8, rw / 8, 2, 1);

   float theta = -atan2f(cos2theta, sin2theta) / 2; // 连续的
//    float theta = roundf
   float Lambda =  8; //roundf(bilinearInterplation(fre, 16, outx, outy, rh /16, rw/16, 1, 0));

   convolution(data, theta, Lambda, rescos, ressin, resenh);

}



'''

EP = SourceModule(code).get_function('EnhanceImageProduce')
stream = cuda.Stream(np.long(torch.cuda.default_stream().cuda_stream))


def cuGabor(images, orifield, fre=None, stream=stream):
    b, _, h, w = images.shape
    assert h % 32 == 0 and w % 32 == 0, "32!"

    images = torch.nn.functional.pad(images, (12, 12, 12, 12))
    pha = torch.empty((b, 1, h, w), dtype=torch.float32, device=torch.device(0))
    amp = torch.empty((b, 1, h, w), dtype=torch.float32, device=torch.device(0))
    enhI = torch.empty((b, 1, h, w), dtype=torch.float32, device=torch.device(0))

    cuPha = np.uintp(pha.data_ptr())
    cuAmp = np.uintp(amp.data_ptr())
    cuEnh = np.uintp(enhI.data_ptr())
    cuOri = np.uintp(orifield.data_ptr())
    images = np.uintp(images.data_ptr())
    cuFre = np.uintp(fre.data_ptr()) if fre is not None else cuOri

    EP.prepare('PPPPPP')
    EP.prepared_async_call((h//32, w //32, b), (32, 32, 1), stream, int(images), int(cuOri), int(cuFre), int(cuPha),int(cuAmp), int(cuEnh))

    return pha, amp, enhI
    