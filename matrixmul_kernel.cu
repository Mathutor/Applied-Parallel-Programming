/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if( col < MATRIX_SIZE && row < MATRIX_SIZE) 
    {
        for(int i = 0; i < MATRIX_SIZE; i++) 
        {
            sum += M.elements[row * MATRIX_SIZE + i] * N.elements[i * MATRIX_SIZE + col];
        }
        P.elements[row * MATRIX_SIZE + col] = sum;
    }

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
