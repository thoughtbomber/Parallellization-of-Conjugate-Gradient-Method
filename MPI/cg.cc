#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <mpi.h>


const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

void CGSolver::solve(std::vector<double> & x) {

  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  std::vector<double> r(m_m);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_m);
  std::vector<double> tmp(m_m);
  std::vector<double> pAp_local(psize); 
  std::vector<double> rsold_local(psize); 
  std::vector<double> rsnew_local(psize); 
  double pAp_sent, rsold_sent, rsnew_sent;
  int block_size = ceil(double(m_n)/double(psize));
  
  std::fill_n(Ap.begin(), Ap.size(), 0.);
   
  r = m_b;
 
  MPI_Allgather(r.data(), m_m, MPI_DOUBLE, p.data(), m_m, MPI_DOUBLE, MPI_COMM_WORLD);

  double rsold = 0.0;

  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                p.data(), 1, 0., Ap.data(), 1);
  
    // calculate and Allgather pAp_local
    pAp_sent = cblas_ddot(m_m, p.data() + prank * block_size, 1, Ap.data(), 1); 
    MPI_Allgather(&pAp_sent, 1, MPI_DOUBLE, pAp_local.data() , 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double pAp_sum = std::accumulate(pAp_local.begin(), pAp_local.end(), 0.0);

    // rsold = r' * r;
    rsold_sent = cblas_ddot(m_m, r.data(), 1, r.data(), 1); 
    MPI_Allgather(&rsold_sent, 1, MPI_DOUBLE, rsold_local.data() , 1, MPI_DOUBLE, MPI_COMM_WORLD);
    rsold = std::accumulate(rsold_local.begin(), rsold_local.end(), 0.0);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(pAp_sum, rsold * NEARZERO);
    
    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_m, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
      rsnew_sent = cblas_ddot(m_m, r.data(), 1, r.data(), 1); 
    MPI_Allgather(&rsnew_sent, 1, MPI_DOUBLE, rsnew_local.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double rsnew = std::accumulate(rsnew_local.begin(), rsnew_local.end(), 0.0);


    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;

    // p = r + (rsnew / rsold) * p; Calculate each portion in each process
    tmp = r;
    cblas_daxpy(m_m, beta, p.data() + prank * block_size, 1, tmp.data(), 1);
   
    // allgather them to make sure all the processes have the same p_k+1
    MPI_Allgather(tmp.data(), m_m,  MPI_DOUBLE, p.data(), m_m,  MPI_DOUBLE, MPI_COMM_WORLD); 
    
    // rsold = rsnew;
    rsold = rsnew;

  
    if (DEBUG && prank == 0) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG) {
    std::fill_n(r.begin(), r.size(), 0.);
    // r = Ax
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., r.data(), 1);
    // r = ax - b
    cblas_daxpy(m_m, -1., m_b.data(), 1, r.data(), 1);

    std::vector<double> Axb_local(psize); 
    std::vector<double> bb_local(psize); 
    
    double Axb_sent = cblas_ddot(m_m, r.data(), 1, r.data(), 1);
    double bb_sent = cblas_ddot(m_m, m_b.data(), 1, m_b.data(), 1);

    MPI_Allgather(&Axb_sent, 1, MPI_DOUBLE, Axb_local.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&bb_sent, 1, MPI_DOUBLE, bb_local.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    double Axb = std::accumulate(Axb_local.begin(), Axb_local.end(), 0.0);
    double bb = std::accumulate(bb_local.begin(), bb_local.end(), 0.0);
    // r' * r/ b' * b
    auto res = std::sqrt(Axb) /
               std::sqrt(bb);
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));

    if (prank == 0) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
    }

  }
}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename); // define in matrix.cc
  m_m = m_A.m();
  m_n = m_A.n();

  // // Test code to print the matrix
  // int prank, psize;
  // MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  // MPI_Comm_size(MPI_COMM_WORLD, &psize);
  
  // for(int i = 0; i <m_m; i++){
  //   std::cout<< "We are in the rank "<< prank<< "Line "<<prank*m_m + i + 1<<std::endl; 
  //   for (int j = 0; j<m_n; j++){
  //     std::cout<< m_A(i,j) << "  ";
  //   }
  //   std::cout <<std::endl;
  // }
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_m); //  Resizes the container so that it contains n elements.

  int prank, psize, k;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  
  int block_size = ceil(double(m_n)/double(psize));

  for (int i = 0; i < m_m; i++) {
    k = i + prank * block_size; 
    m_b[i] = -2. * k * M_PI * M_PI * std::sin(10. * M_PI * k * h) *
             std::sin(10. * M_PI * k * h);
  }


  // for (int i = 0; i < m_n; i++) {
  //   m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
  //            std::sin(10. * M_PI * i * h);
  // }
}
