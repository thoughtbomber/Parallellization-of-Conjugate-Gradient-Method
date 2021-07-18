#include "matrix.hh"
#include "matrix_coo.hh"
#include <iostream>
#include <string>
#include <mpi.h>
#include <cmath>

void Matrix::read(const std::string & fn) {

  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  MatrixCOO mat;
  mat.read(fn); // read, in matrix_coo.cc

  int block_size = ceil(double(mat.m())/double(psize));

  if (prank != psize - 1){
    resize(block_size, mat.n()); // define the size of the matrix, in matrix.hh
  }
  else{
    resize(mat.m() - block_size * prank, mat.n());
  }
  

  for (int z = 0; z < mat.nz(); ++z) {
    auto i = mat.irn[z]; // i and j have been rescaled to 0 base.
    auto j = mat.jcn[z];
    auto a = mat.a[z];

   //std::cout<< "Block size " << block_size<<std::endl;
   //std::cout<<"We are working on "<< i << "  " << i % block_size * m_n + j <<"   "<<a<<std::endl;

  
    if (i < block_size * (prank + 1) && i >= block_size * prank){
       m_a[i % block_size * m_n + j] = a;
    }

    if (mat.is_sym() && j < block_size * (prank + 1) && j >= block_size * prank) {
      m_a[j % block_size * m_n + i] = a;
    }
  }
}
