#include "cg.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include "cublas_v2.h"
#include <cuda_runtime.h>
// #include <cblas.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

__global__ void compute_step_one_thread_per_row(Matrix m_A, double *p, double *Ap) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
   
    for (int j = 0; j < m_A.n(); j++) {
        // computation of the new step
        sum += m_A(i, j) * p[j];
    }
    Ap[i] = sum;
    
}

__global__ void my_daxpy(double*x, double *p, double alpha, double beta) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   x[i] = alpha * x[i] + beta * p[i];
    
}

__global__ void my_daxpy_altered_order(double*x, double *p, double alpha, double beta) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   x[i] = alpha * p[i] + beta * x[i];
    
}



void CGSolver::solve(std::vector<double> & x_input, int block_size) {
  //std::vector<double> r(m_n);
  //std::vector<double> p(m_n);
  //std::vector<double> Ap(m_n);
  //std::vector<double> tmp(m_n);

  double* r{nullptr};
  double* p{nullptr};
  double* Ap{nullptr};
  double* tmp{nullptr};
  double* x{nullptr};
  cudaMallocManaged(&r, m_n * sizeof(double));
  cudaMallocManaged(&p, m_n * sizeof(double));
  cudaMallocManaged(&Ap, m_n * sizeof(double));
  cudaMallocManaged(&tmp, m_n * sizeof(double));
  cudaMallocManaged(&x, m_n * sizeof(double));

  for (int i = 0; i < m_n; i++) x[i] = x_input[i];

  int grid_size = m_m/block_size;
  std::cout<<"The grid size is "<< grid_size << ", and the block size is "<< block_size<<std::endl;
  cublasHandle_t handle;
	cublasCreate(&handle);

  // r = b - A * x;
  std::fill_n(Ap, m_n, 0.);

  for(int i = 0; i < m_n; i++) r[i] = m_b[i];
  // cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);
  // p = r;
  for(int i = 0; i < m_n; i++) p[i] = r[i];

  // rsold = r' * r;
  //  auto rsold = cblas_ddot(m_n, r.data(), 1, p.data(), 1);
  double rsold;
  cublasDdot(handle, m_n, r, 1, p, 1, &rsold);


  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    std::fill_n(Ap, m_n, 0.);

    // Parallel the matrix-vector product 
    dim3 dimGrid_per_row(grid_size);
    dim3 dimBlock_per_row(block_size);
    compute_step_one_thread_per_row <<<grid_size, block_size>>> (m_A, p, Ap);
    cudaDeviceSynchronize();



    // alpha = rsold / (p' * Ap);
    double pAp;
    cublasDdot(handle, m_n, p,1, Ap, 1, &pAp);
    auto alpha =  rsold / std::max(pAp, rsold * NEARZERO);
    cudaDeviceSynchronize();
    

    // x = x + alpha * p;
    // cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);
    my_daxpy <<<grid_size, block_size>>> (x, p, 1.0, alpha);
    
    
    
    // r = r - alpha * Ap;
    // cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);
    my_daxpy <<<grid_size, block_size>>> (r, Ap, 1.0, -alpha);
    cudaDeviceSynchronize();
  
  
    
  
    // rsnew = r' * r;
    // auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);
    double rsnew;
    cublasDdot(handle, m_n, r, 1, r, 1, &rsnew);
    cudaDeviceSynchronize();
    


    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    // cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    my_daxpy_altered_order <<<grid_size, block_size>>> (p, r, 1.0, beta);
    cudaDeviceSynchronize();


    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG) {
    std::fill_n(r, m_n, 0.);
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
    //            x.data(), 1, 0., r.data(), 1);
    compute_step_one_thread_per_row <<<grid_size, block_size>>> (m_A, x, r);
    cudaDeviceSynchronize();

    // cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);   
    my_daxpy <<<grid_size, block_size>>> (r, m_b, 1.0, -1.0);
    cudaDeviceSynchronize();
  
    
   // auto res = std::sqrt(cblasDdot(m_n, r.data(), 1, r.data(), 1)) /
   //             std::sqrt(cblasDdot(m_n, m_b.data(), 1, m_b.data(), 1));
    double norminator, denominator;
    cublasDdot(handle, m_n, r, 1, r, 1, &norminator);
    cublasDdot(handle, m_n, m_b, 1, m_b, 1, &denominator);
    auto res = norminator/denominator;

   // auto nx = std::sqrt(cblasDdot(m_n, x.data(), 1, x.data(), 1));
    double squared_nx;
    cublasDdot(handle, m_n, x, 1, x, 1, &squared_nx);
    auto nx = std::sqrt(squared_nx);
   // for(int i = 0; i< 10; i++) std::cout<<m_b[i]<< "  b"<<std::endl;
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }

  cudaFree(r);
  cudaFree(p);
  cudaFree(Ap);
  cudaFree(tmp);
  cudaFree(x);
  cublasDestroy(handle);

}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

void Solver::init_source_term(double h) {
  resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
