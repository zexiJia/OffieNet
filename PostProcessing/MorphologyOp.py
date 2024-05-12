
import torch
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from os import path
from pycuda.driver import Stream
import math as m
magic_code = torch.cuda.FloatTensor(8)

code = r'''

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>


// 要修改 这三个值都要改
#define shared_size 50 // max padding is (52 - 32) / 2 = 10


__device__ bool erode(bool *data, int r){
    bool res = true;
    for(int i = -r ; i <= r; i++ ){
        for(int j = -r ; j <= r; j++){       
            bool d = *(data + (threadIdx.x + i + r) * shared_size + threadIdx.y + j + r) ;
            res = res && d;
        }
    }
    return res;
}


__device__ bool dilate(bool *data, int r){
    bool res = false;
    for(int i = -r ; i <= r; i++ ){
        for(int j = -r ; j <= r; j++){       
            bool d = *(data + (threadIdx.x + i + r) * shared_size + threadIdx.y + j + r) ;
            res = res || d;
        }
    }
    return res;
}

__device__ void Pad(bool *data, bool *src, int batch, int h, int w, int r=3){
    for(int i = threadIdx.x; i <= blockDim.x + 2 * r; i+=blockDim.x ){
        for(int j = threadIdx.y; j <= blockDim.x + 2 * r; j+=blockDim.y){
            int x = blockIdx.x*blockDim.x + i - r;
            int y = blockIdx.y*blockDim.y + j - r;
            if(x>=0 && y>= 0 && x< h && y < w){
                *(data+ i*shared_size + j) = *(src + batch*h*w + x * w + y);
            }else{
                *(data+ i*shared_size + j) = false;
            }    
        }
    }
    __syncthreads();
}



__global__ void Dilate(bool *segment, bool * resSegment, int *height, int *width, int *radio){

    int batch = blockIdx.z;
    int outx = blockIdx.x * blockDim.x + threadIdx.x;
    int outy = blockIdx.y * blockDim.y + threadIdx.y;

    int h = (*height);
    int w = (*width) ;
    int r = (*radio);

    __shared__ bool data[shared_size * shared_size];
    __syncthreads();
    Pad(data, segment, batch, h, w, r);
    if(outx < h && outy < w){
        *(resSegment +  batch * w * h + outx * w + outy) = dilate(data, r);
    } 
}



__global__ void Erode(bool *segment, bool * resSegment, int *height, int *width, int *radio){

    int batch = blockIdx.z;
    int outx = blockIdx.x * blockDim.x + threadIdx.x;
    int outy = blockIdx.y * blockDim.y + threadIdx.y;

    int h = (*height);
    int w = (*width);
    int r = (*radio);

    __shared__ bool data[shared_size * shared_size];
    Pad(data, segment, batch, h, w, r);
    if(outx < h && outy < w){
        *(resSegment +  batch * w * h + outx * w + outy) = erode(data, r);
    }

}




__global__ void OpenOp(bool *segment, bool * MiddleRes, bool * resSegment, int *height, int *width, int *radio){
    
    int batch = blockIdx.z;
    int outx = blockIdx.x * blockDim.x + threadIdx.x;
    int outy = blockIdx.y * blockDim.y + threadIdx.y;

    int h = (*height);
    int w = (*width);
    int r = (*radio);

    __shared__ bool data[shared_size * shared_size];
    Pad(data, segment, batch, h, w, r);
    if(outx < h && outy < w){
        *(MiddleRes +  batch * w * h + outx * w + outy) = erode(data, r);
    }

    __threadfence();
    Pad(data, MiddleRes, batch, h, w, r);
    if(outx < h && outy < w){
        *(resSegment +  batch * w * h + outx * w + outy) = dilate(data, r);
    }

}



__global__ void CloseOp(bool *segment, bool * MiddleRes, bool * resSegment, int *height, int *width, int *radio){
    
    int batch = blockIdx.z;
    int outx = blockIdx.x * blockDim.x + threadIdx.x;
    int outy = blockIdx.y * blockDim.y + threadIdx.y;

    int h = (*height);
    int w = (*width);
    int r = (*radio);

    __shared__ bool data[shared_size * shared_size];
    Pad(data, segment, batch, h, w, r);
    if(outx < h && outy < w){
        *(MiddleRes +  batch * w * h + outx * w + outy) = dilate(data, r);
    }

    __threadfence();
    Pad(data, MiddleRes, batch, h, w, r);
    if(outx < h && outy < w){
        *(resSegment +  batch * w * h + outx * w + outy) = erode(data, r);
    }

}







'''

Dilate = SourceModule(code).get_function('Dilate')
Erode = SourceModule(code).get_function('Erode')
OpenOp = SourceModule(code).get_function('OpenOp')
CloseOp = SourceModule(code).get_function('CloseOp')
stream = cuda.Stream(np.long(torch.cuda.default_stream().cuda_stream))
Dilate.prepare('PPPPP')
Erode.prepare('PPPPP')
OpenOp.prepare('PPPPPP')
CloseOp.prepare('PPPPPP')



def cuDilate(segment, radio=5, stream=stream):
    b, _, h, w = segment.shape
    resSeg = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))

    cuSeg = np.uintp(resSeg.data_ptr())
    segment = np.uintp(segment.data_ptr())
    hh = cuda.to_device(np.int32(h))
    ww = cuda.to_device(np.int32(w))
    radio = cuda.to_device(np.int32(radio))
    Dilate.prepared_async_call((m.ceil(h/32), m.ceil(w/32), b), (32, 32, 1), stream, int(segment), int(cuSeg), int(hh), int(ww), int(radio))
    
    return resSeg


def cuErode(segment, radio=5, stream=stream):
    b, _, h, w = segment.shape
    resSeg = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))

    cuSeg = np.uintp(resSeg.data_ptr())
    segment = np.uintp(segment.data_ptr())
    hh = cuda.to_device(np.int32(h))
    ww = cuda.to_device(np.int32(w))
    radio = cuda.to_device(np.int32(radio))
    Erode.prepared_async_call((m.ceil(h/32), m.ceil(w/32), b), (32, 32, 1), stream, int(segment), int(cuSeg), int(hh), int(ww), int(radio))
    return resSeg



def cuOpenOp(segment, radio=5,  stream=stream):
    b, _, h, w = segment.shape

    Middle = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))
    resSeg = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))

    cuSeg = np.uintp(resSeg.data_ptr())
    cuMiddle = np.uintp(Middle.data_ptr())
    segment = np.uintp(segment.data_ptr())
    hh = cuda.to_device(np.int32(h))
    ww = cuda.to_device(np.int32(w))
    r = cuda.to_device(np.int32(radio))
 
    OpenOp.prepared_async_call((m.ceil(h/32), m.ceil(w/32), b), (32, 32, 1), stream, int(segment), int(cuMiddle), int(cuSeg), int(hh), int(ww), int(r)) 
    return resSeg


def cuCloseOp(segment, radio=5, stream=stream):
    b, _, h, w = segment.shape

    Middle = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))
    resSeg = torch.zeros((b, 1, h, w), dtype=torch.bool, device=torch.device(0))

    cuSeg = np.uintp(resSeg.data_ptr())
    cuMiddle = np.uintp(Middle.data_ptr())
    segment = np.uintp(segment.data_ptr())
    hh = cuda.to_device(np.int32(h))
    ww = cuda.to_device(np.int32(w))
    r = cuda.to_device(np.int32(radio))
 
    CloseOp.prepared_async_call((m.ceil(h/32), m.ceil(w/32), b), (32, 32, 1), stream, int(segment), int(cuMiddle), int(cuSeg), int(hh), int(ww), int(r)) 
    return resSeg