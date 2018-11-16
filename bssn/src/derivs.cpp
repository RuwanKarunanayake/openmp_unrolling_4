#include <cmath>
#include <iostream>
#include "derivs.h"
#include <omp.h>


/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_x(double * const  Dxu, const double * const  u,
               const double dx, const unsigned int *sz, unsigned bflag)
{
  const double idx = 1.0/dx;
  const double idx_by_2 = 0.5 * idx;
  const double idx_by_12 = idx / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 1;
  const int kb = 1;
  const int ie = sz[0]-3;
  const int je = sz[1]-1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  const int ke = sz[2]-1;
    const int n=1;
    
  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        Dxu[pp] = (u[pp-2] -8.0*u[pp-1] + 8.0*u[pp+1] - u[pp+2] ) * idx_by_12;
        
        Dxu[pp+1] = (u[pp-1] -8.0*u[pp] + 8.0*u[pp+2] - u[pp+3] ) * idx_by_12;
        
        Dxu[pp+2] = (u[pp] -8.0*u[pp+1] + 8.0*u[pp+3] - u[pp+4] ) * idx_by_12;
        
        Dxu[pp+3] = (u[pp+1] -8.0*u[pp+2] + 8.0*u[pp+4] - u[pp+5] ) * idx_by_12;
        
        // Dxu[pp+4] = (u[pp+2] -8.0*u[pp+3] + 8.0*u[pp+5] - u[pp+6] ) * idx_by_12;
        
        // Dxu[pp+5] = (u[pp+3] -8.0*u[pp+4] + 8.0*u[pp+6] - u[pp+7] ) * idx_by_12;
        
        // Dxu[pp+6] = (u[pp+4] -8.0*u[pp+5] + 8.0*u[pp+7] - u[pp+8] ) * idx_by_12;
        
        // Dxu[pp+7] = (u[pp+5] -8.0*u[pp+6] + 8.0*u[pp+8] - u[pp+9] ) * idx_by_12;
      }
      
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        Dxu[pp] = (u[pp-2] -8.0*u[pp-1] + 8.0*u[pp+1] - u[pp+2] ) * idx_by_12;
      }
    }
  }


  if (bflag & (1u<<OCT_DIR_LEFT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        Dxu[IDX(3,j,k)] = ( -  3.0 * u[IDX(3,j,k)]
                            +  4.0 * u[IDX(4,j,k)]
                            -        u[IDX(5,j,k)]
                          ) * idx_by_2;
        Dxu[IDX(4,j,k)] = ( - u[IDX(3,j,k)]
                            + u[IDX(5,j,k)]
                          ) * idx_by_2;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_RIGHT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        Dxu[IDX(ie-2,j,k)] = ( - u[IDX(ie-3,j,k)]
                               + u[IDX(ie-1,j,k)]
                             ) * idx_by_2;

        Dxu[IDX(ie-1,j,k)] = (        u[IDX(ie-3,j,k)]
                              - 4.0 * u[IDX(ie-2,j,k)]
                              + 3.0 * u[IDX(ie-1,j,k)]
                             ) * idx_by_2;

      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
#pragma message("DEBUG_DERIVS_COMP: ON")
  for (int k = 3; k < sz[2]-3; k++) {
    for (int j = 3; j < sz[1]-3; j++) {
      for (int i = 3; i < sz[0]-3; i++) {
        int pp = IDX(i,j,k);
         if(isnan(Dxu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif


}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_y(double * const  Dyu, const double * const  u,
               const double dy, const unsigned int *sz, unsigned bflag)
{
  const double idy = 1.0/dy;
  const double idy_by_2 = 0.50 * idy;
  const double idy_by_12 = idy / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 1;
  const int ie = sz[0]-3;
  const int je = sz[1]-3;
  const int ke = sz[2]-1;

    const int n=nx;
  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        Dyu[pp] = (u[pp-2*nx] - 8.0*u[pp-nx] + 8.0*u[pp+nx] - u[pp+2*nx])*idy_by_12;
        
        Dyu[pp+1] = (u[pp+1-2*nx] - 8.0*u[pp+1-nx] + 8.0*u[pp+1+nx] - u[pp+1+2*nx])*idy_by_12;
        
        Dyu[pp+2] = (u[pp+2-2*nx] - 8.0*u[pp+2-nx] + 8.0*u[pp+2+nx] - u[pp+2+2*nx])*idy_by_12;
        
        Dyu[pp+3] = (u[pp+3-2*nx] - 8.0*u[pp+3-nx] + 8.0*u[pp+3+nx] - u[pp+3+2*nx])*idy_by_12;
        
        // Dyu[pp+4] = (u[pp+4-2*nx] - 8.0*u[pp+4-nx] + 8.0*u[pp+4+nx] - u[pp+4+2*nx])*idy_by_12;
        
        // Dyu[pp+5] = (u[pp+5-2*nx] - 8.0*u[pp+5-nx] + 8.0*u[pp+5+nx] - u[pp+5+2*nx])*idy_by_12;
        
        // Dyu[pp+6] = (u[pp+6-2*nx] - 8.0*u[pp+6-nx] + 8.0*u[pp+6+nx] - u[pp+6+2*nx])*idy_by_12;
        
        // Dyu[pp+7] = (u[pp+7-2*nx] - 8.0*u[pp+7-nx] + 8.0*u[pp+7+nx] - u[pp+7+2*nx])*idy_by_12;
      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        Dyu[pp] = (u[pp-2*nx] - 8.0*u[pp-nx] + 8.0*u[pp+nx] - u[pp+2*nx])*idy_by_12;
      }
    }
  }


  if (bflag & (1u<<OCT_DIR_DOWN)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        Dyu[IDX(i, 3,k)] = ( - 3.0 * u[IDX(i,3,k)]
                            +  4.0 * u[IDX(i,4,k)]
                            -        u[IDX(i,5,k)]
                          ) * idy_by_2;

        Dyu[IDX(i,4,k)] = ( - u[IDX(i,3,k)]
                            + u[IDX(i,5,k)]
                          ) * idy_by_2;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_UP)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        Dyu[IDX(i,je-2,k)] = ( - u[IDX(i,je-3,k)]
                               + u[IDX(i,je-1,k)]
                             ) * idy_by_2;

        Dyu[IDX(i,je-1,k)] = (        u[IDX(i,je-3,k)]
                              - 4.0 * u[IDX(i,je-2,k)]
                              + 3.0 * u[IDX(i,je-1,k)]
                          ) * idy_by_2;
      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = 3; k < sz[2]-3; k++) {
    for (int j = 3; j < sz[1]-3; j++) {
      for (int i = 3; i < sz[0]-3; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Dyu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_z(double * const  Dzu, const double * const  u,
               const double dz, const unsigned int *sz, unsigned bflag)
{
  const double idz = 1.0/dz;
  const double idz_by_2 = 0.50 * idz;
  const double idz_by_12 = idz / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0]-3;
  const int je = sz[1]-3;
  const int ke = sz[2]-3;

  const int n = nx*ny;
  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        Dzu[pp] = (u[pp-2*n] - 8.0*u[pp-n] + 8.0*u[pp+n] - u[pp+2*n]) * idz_by_12;
        
        Dzu[pp+1] = (u[pp+1-2*n] - 8.0*u[pp+1-n] + 8.0*u[pp+1+n] - u[pp+1+2*n]) * idz_by_12;
        
        Dzu[pp+2] = (u[pp+2-2*n] - 8.0*u[pp+2-n] + 8.0*u[pp+2+n] - u[pp+2+2*n]) * idz_by_12;
        
        Dzu[pp+3] = (u[pp+3-2*n] - 8.0*u[pp+3-n] + 8.0*u[pp+3+n] - u[pp+3+2*n]) * idz_by_12;
        
        // Dzu[pp+4] = (u[pp+4-2*n] - 8.0*u[pp+4-n] + 8.0*u[pp+4+n] - u[pp+4+2*n]) * idz_by_12;
        
        // Dzu[pp+5] = (u[pp+5-2*n] - 8.0*u[pp+5-n] + 8.0*u[pp+5+n] - u[pp+5+2*n]) * idz_by_12;
        
        // Dzu[pp+6] = (u[pp+6-2*n] - 8.0*u[pp+6-n] + 8.0*u[pp+6+n] - u[pp+6+2*n]) * idz_by_12;
        
        // Dzu[pp+7] = (u[pp+7-2*n] - 8.0*u[pp+7-n] + 8.0*u[pp+7+n] - u[pp+7+2*n]) * idz_by_12;
      }
    }
  }


  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        Dzu[pp] = (u[pp-2*n] - 8.0*u[pp-n] + 8.0*u[pp+n] - u[pp+2*n]) * idz_by_12;
      }
    }
  } 

  if (bflag & (1u<<OCT_DIR_BACK)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        Dzu[IDX(i, j, 3)] = ( - 3.0 *  u[IDX(i,j,3)]
                              +  4.0 * u[IDX(i,j,4)]
                              -        u[IDX(i,j,5)]
                            ) * idz_by_2;

        Dzu[IDX(i,j,4)] = ( - u[IDX(i,j,3)]
                            + u[IDX(i,j,5)]
                          ) * idz_by_2;

      }
    }
  }

  if (bflag & (1u<<OCT_DIR_FRONT)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        Dzu[IDX(i,j,ke-2)] = ( - u[IDX(i,j,ke-3)]
                               + u[IDX(i,j,ke-1)]
                             ) * idz_by_2;

        Dzu[IDX(i,j,ke-1)] = (        u[IDX(i,j,ke-3)]
                              - 4.0 * u[IDX(i,j,ke-2)]
                              + 3.0 * u[IDX(i,j,ke-1)]
                             ) * idz_by_2;

      }
    }

  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Dzu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_xx(double * const  DxDxu, const double * const  u,
                const double dx, const unsigned int *sz, unsigned bflag)
{

  const double idx_sqrd = 1.0/(dx*dx);
  const double idx_sqrd_by_12 = idx_sqrd / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        DxDxu[pp] = (   -        u[pp-2]
                        + 16.0 * u[pp-1]
                        - 30.0 * u[pp]
                        + 16.0 * u[pp+1]
                        -        u[pp+2]
                    ) * idx_sqrd_by_12;
        
        DxDxu[pp+1] = (   -        u[pp-1]
                        + 16.0 * u[pp]
                        - 30.0 * u[pp+1]
                        + 16.0 * u[pp+2]
                        -        u[pp+3]
                    ) * idx_sqrd_by_12;
        
        DxDxu[pp+2] = (   -        u[pp]
                        + 16.0 * u[pp+1]
                        - 30.0 * u[pp+2]
                        + 16.0 * u[pp+3]
                        -        u[pp+4]
                    ) * idx_sqrd_by_12;
        
        DxDxu[pp+3] = (   -        u[pp+1]
                        + 16.0 * u[pp+2]
                        - 30.0 * u[pp+3]
                        + 16.0 * u[pp+4]
                        -        u[pp+5]
                    ) * idx_sqrd_by_12;
        
        // DxDxu[pp+4] = (   -        u[pp+2]
        //                 + 16.0 * u[pp+3]
        //                 - 30.0 * u[pp+4]
        //                 + 16.0 * u[pp+5]
        //                 -        u[pp+6]
        //             ) * idx_sqrd_by_12;
        
        // DxDxu[pp+5] = (   -        u[pp+3]
        //                 + 16.0 * u[pp+4]
        //                 - 30.0 * u[pp+5]
        //                 + 16.0 * u[pp+6]
        //                 -        u[pp+7]
        //             ) * idx_sqrd_by_12;
        
        // DxDxu[pp+6] = (   -        u[pp+4]
        //                 + 16.0 * u[pp+5]
        //                 - 30.0 * u[pp+6]
        //                 + 16.0 * u[pp+7]
        //                 -        u[pp+8]
        //             ) * idx_sqrd_by_12;
        
        // DxDxu[pp+7] = (   -        u[pp+5]
        //                 + 16.0 * u[pp+6]
        //                 - 30.0 * u[pp+7]
        //                 + 16.0 * u[pp+8]
        //                 -        u[pp+9]
        //             ) * idx_sqrd_by_12;

      }
    }
  }


  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        DxDxu[pp] = (   -        u[pp-2]
                        + 16.0 * u[pp-1]
                        - 30.0 * u[pp]
                        + 16.0 * u[pp+1]
                        -        u[pp+2]
                    ) * idx_sqrd_by_12;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_LEFT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        DxDxu[IDX(3,j,k)] = (   2.0 * u[IDX(3,j,k)]
                              - 5.0 * u[IDX(4,j,k)]
                              + 4.0 * u[IDX(5,j,k)]
                              -       u[IDX(6,j,k)]
                            ) * idx_sqrd;

        DxDxu[IDX(4,j,k)] = (         u[IDX(3,j,k)]
                              - 2.0 * u[IDX(4,j,k)]
                              +       u[IDX(5,j,k)]
                            ) * idx_sqrd;

      }
    }
  }

  if (bflag & (1u<<OCT_DIR_RIGHT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        DxDxu[IDX(ie-2,j,k)] = (       u[IDX(ie-3,j,k)]
                               - 2.0 * u[IDX(ie-2,j,k)]
                               +       u[IDX(ie-1,j,k)]
                              ) * idx_sqrd;

        DxDxu[IDX(ie-1,j,k)] = ( -    u[IDX(ie-4,j,k)]
                          + 4.0 * u[IDX(ie-3,j,k)]
                          - 5.0 * u[IDX(ie-2,j,k)]
                          + 2.0 * u[IDX(ie-1,j,k)]
                        ) * idx_sqrd;

      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(DxDxu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_yy(double * const  DyDyu, const double * const  u,
                const double dy, const unsigned int *sz, unsigned bflag)
{

  const double idy_sqrd = 1.0/(dy*dy);
  const double idy_sqrd_by_12 = idy_sqrd / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        DyDyu[pp] = ( -u[pp-2*nx] + 16.0 * u[pp-nx] - 30.0 * u[pp]
                               + 16.0 * u[pp+nx] - u[pp+2*nx]
                 ) * idy_sqrd_by_12;
        
        DyDyu[pp+1] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
                               + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
                 ) * idy_sqrd_by_12;
        
        DyDyu[pp+2] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
                               + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
                 ) * idy_sqrd_by_12;
        
        DyDyu[pp+3] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
                               + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
                 ) * idy_sqrd_by_12;
        
        // DyDyu[pp+4] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
        //                        + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
        //          ) * idy_sqrd_by_12;
        
        // DyDyu[pp+5] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
        //                        + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
        //          ) * idy_sqrd_by_12;
        
        // DyDyu[pp+6] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
        //                        + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
        //          ) * idy_sqrd_by_12;
        
        // DyDyu[pp+7] = ( -u[pp+1-2*nx] + 16.0 * u[pp+1-nx] - 30.0 * u[pp+1]
        //                        + 16.0 * u[pp+1+nx] - u[pp+1+2*nx]
        //          ) * idy_sqrd_by_12;

      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        DyDyu[pp] = ( -u[pp-2*nx] + 16.0 * u[pp-nx] - 30.0 * u[pp]
                               + 16.0 * u[pp+nx] - u[pp+2*nx]
                 ) * idy_sqrd_by_12;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_DOWN)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        DyDyu[IDX(i,3,k)] = (   2.0 * u[IDX(i,3,k)]
                           - 5.0 * u[IDX(i,4,k)]
                           + 4.0 * u[IDX(i,5,k)]
                           -       u[IDX(i,6,k)]
                        ) * idy_sqrd;

        DyDyu[IDX(i,4,k)] = (         u[IDX(i,3,k)]
                           - 2.0 * u[IDX(i,4,k)]
                           +       u[IDX(i,5,k)]
                        ) * idy_sqrd;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_UP)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        DyDyu[IDX(i,je-2,k)] = (      u[IDX(i,je-3,k)]
                           - 2.0 * u[IDX(i,je-2,k)]
                           +       u[IDX(i,je-1,k)]
                          ) * idy_sqrd;

        DyDyu[IDX(i,je-1,k)] = ( -   u[IDX(i,je-4,k)]
                          + 4.0 * u[IDX(i,je-3,k)]
                          - 5.0 * u[IDX(i,je-2,k)]
                          + 2.0 * u[IDX(i,je-1,k)]
                        ) * idy_sqrd;

      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(DyDyu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42_zz(double * const  DzDzu, const double * const  u,
                const double dz, const unsigned int *sz, unsigned bflag)
{

  const double idz_sqrd = 1.0/(dz*dz);
  const double idz_sqrd_by_12 = idz_sqrd / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  const int n = nx * ny;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        DzDzu[pp] = ( - u[pp-2*n] + 16.0 * u[pp-n] - 30.0 * u[pp]
                   + 16.0 * u[pp+n] - u[pp+2*n] ) * idz_sqrd_by_12;
        
        DzDzu[pp+1] = ( - u[pp+1-2*n] + 16.0 * u[pp+1-n] - 30.0 * u[pp+1]
                   + 16.0 * u[pp+1+n] - u[pp+1+2*n] ) * idz_sqrd_by_12;
        
        DzDzu[pp+2] = ( - u[pp+2-2*n] + 16.0 * u[pp+2-n] - 30.0 * u[pp+2]
                   + 16.0 * u[pp+2+n] - u[pp+2+2*n] ) * idz_sqrd_by_12;
        
        DzDzu[pp+3] = ( - u[pp+3-2*n] + 16.0 * u[pp+3-n] - 30.0 * u[pp+3]
                   + 16.0 * u[pp+3+n] - u[pp+3+2*n] ) * idz_sqrd_by_12;
        
        // DzDzu[pp+4] = ( - u[pp+4-2*n] + 16.0 * u[pp+4-n] - 30.0 * u[pp+4]
        //            + 16.0 * u[pp+4+n] - u[pp+4+2*n] ) * idz_sqrd_by_12;
        
        // DzDzu[pp+5] = ( - u[pp+5-2*n] + 16.0 * u[pp+5-n] - 30.0 * u[pp+5]
        //            + 16.0 * u[pp+5+n] - u[pp+5+2*n] ) * idz_sqrd_by_12;
        
        // DzDzu[pp+6] = ( - u[pp+6-2*n] + 16.0 * u[pp+6-n] - 30.0 * u[pp+6]
        //            + 16.0 * u[pp+6+n] - u[pp+6+2*n] ) * idz_sqrd_by_12;
        
        // DzDzu[pp+7] = ( - u[pp+7-2*n] + 16.0 * u[pp+7-n] - 30.0 * u[pp+7]
        //            + 16.0 * u[pp+7+n] - u[pp+7+2*n] ) * idz_sqrd_by_12;

      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        DzDzu[pp] = ( - u[pp-2*n] + 16.0 * u[pp-n] - 30.0 * u[pp]
                   + 16.0 * u[pp+n] - u[pp+2*n] ) * idz_sqrd_by_12;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_BACK)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        DzDzu[IDX(i,j,3)] = (   2.0 * u[IDX(i,j,3)]
                           - 5.0 * u[IDX(i,j,4)]
                           + 4.0 * u[IDX(i,j,5)]
                           -       u[IDX(i,j,6)]
                        ) * idz_sqrd;

        DzDzu[IDX(i,j,4)] = (         u[IDX(i,j,3)]
                           - 2.0 * u[IDX(i,j,4)]
                           +       u[IDX(i,j,5)]
                        ) * idz_sqrd;

      }
    }
  }

  if (bflag & (1u<<OCT_DIR_FRONT)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        DzDzu[IDX(i,j,ke-2)] = (      u[IDX(i,j,ke-3)]
                           - 2.0 * u[IDX(i,j,ke-2)]
                           +       u[IDX(i,j,ke-1)]
                          ) * idz_sqrd;

        DzDzu[IDX(i,j,ke-1)] = ( -   u[IDX(i,j,ke-4)]
                          + 4.0 * u[IDX(i,j,ke-3)]
                          - 5.0 * u[IDX(i,j,ke-2)]
                          + 2.0 * u[IDX(i,j,ke-1)]
                        ) * idz_sqrd;

      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(DzDzu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}


/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42adv_x(double * const  Dxu, const double * const  u,
                  const double dx, const unsigned int *sz,
                  const double * const betax, unsigned bflag)
{

  const double idx = 1.0/dx;
  const double idx_by_2 = 0.50 * idx;
  const double idx_by_12 = idx / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        if (betax[pp] > 0.0 ) {
          Dxu[pp] = ( -  3.0 * u[pp-1]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+1]
                      -  6.0 * u[pp+2]
                      +        u[pp+3]
                    ) * idx_by_12;
        }
        else {
          Dxu[pp] = ( -        u[pp-3]
                      +  6.0 * u[pp-2]
                      - 18.0 * u[pp-1]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+1]
                    ) * idx_by_12;
        }
        
        if (betax[pp+1] > 0.0 ) {
          Dxu[pp+1] = ( -  3.0 * u[pp]
                      - 10.0 * u[pp+1]
                      + 18.0 * u[pp+2]
                      -  6.0 * u[pp+3]
                      +        u[pp+4]
                    ) * idx_by_12;
        }
        else {
          Dxu[pp+1] = ( -        u[pp-2]
                      +  6.0 * u[pp-1]
                      - 18.0 * u[pp]
                      + 10.0 * u[pp+1]
                      +  3.0 * u[pp+2]
                    ) * idx_by_12;
        }
        
        if (betax[pp+2] > 0.0 ) {
          Dxu[pp+2] = ( -  3.0 * u[pp+1]
                      - 10.0 * u[pp+2]
                      + 18.0 * u[pp+3]
                      -  6.0 * u[pp+4]
                      +        u[pp+5]
                    ) * idx_by_12;
        }
        else {
          Dxu[pp+2] = ( -        u[pp-1]
                      +  6.0 * u[pp]
                      - 18.0 * u[pp+1]
                      + 10.0 * u[pp+2]
                      +  3.0 * u[pp+3]
                    ) * idx_by_12;
        }
        
        if (betax[pp+3] > 0.0 ) {
          Dxu[pp+3] = ( -  3.0 * u[pp+2]
                      - 10.0 * u[pp+3]
                      + 18.0 * u[pp+4]
                      -  6.0 * u[pp+5]
                      +        u[pp+6]
                    ) * idx_by_12;
        }
        else {
          Dxu[pp+3] = ( -        u[pp]
                      +  6.0 * u[pp+1]
                      - 18.0 * u[pp+2]
                      + 10.0 * u[pp+3]
                      +  3.0 * u[pp+4]
                    ) * idx_by_12;
        }
        
        // if (betax[pp+4] > 0.0 ) {
        //   Dxu[pp+4] = ( -  3.0 * u[pp-1]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+1]
        //               -  6.0 * u[pp+2]
        //               +        u[pp+3]
        //             ) * idx_by_12;
        // }
        // else {
        //   Dxu[pp+4] = ( -        u[pp-3]
        //               +  6.0 * u[pp-2]
        //               - 18.0 * u[pp-1]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+1]
        //             ) * idx_by_12;
        // }
        
        // if (betax[pp+5] > 0.0 ) {
        //   Dxu[pp+5] = ( -  3.0 * u[pp-1]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+1]
        //               -  6.0 * u[pp+2]
        //               +        u[pp+3]
        //             ) * idx_by_12;
        // }
        // else {
        //   Dxu[pp+5] = ( -        u[pp-3]
        //               +  6.0 * u[pp-2]
        //               - 18.0 * u[pp-1]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+1]
        //             ) * idx_by_12;
        // }
        
        // if (betax[pp+6] > 0.0 ) {
        //   Dxu[pp+6] = ( -  3.0 * u[pp-1]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+1]
        //               -  6.0 * u[pp+2]
        //               +        u[pp+3]
        //             ) * idx_by_12;
        // }
        // else {
        //   Dxu[pp+6] = ( -        u[pp-3]
        //               +  6.0 * u[pp-2]
        //               - 18.0 * u[pp-1]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+1]
        //             ) * idx_by_12;
        // }
        
        // if (betax[pp+7] > 0.0 ) {
        //   Dxu[pp+7] = ( -  3.0 * u[pp-1]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+1]
        //               -  6.0 * u[pp+2]
        //               +        u[pp+3]
        //             ) * idx_by_12;
        // }
        // else {
        //   Dxu[pp+7] = ( -        u[pp-3]
        //               +  6.0 * u[pp-2]
        //               - 18.0 * u[pp-1]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+1]
        //             ) * idx_by_12;
        // }

      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        if (betax[pp] > 0.0 ) {
          Dxu[pp] = ( -  3.0 * u[pp-1]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+1]
                      -  6.0 * u[pp+2]
                      +        u[pp+3]
                    ) * idx_by_12;
        }
        else {
          Dxu[pp] = ( -        u[pp-3]
                      +  6.0 * u[pp-2]
                      - 18.0 * u[pp-1]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+1]
                    ) * idx_by_12;
        }
      }
    }
  }


  if (bflag & (1u<<OCT_DIR_LEFT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        Dxu[IDX(3,j,k)] = ( -  3.0 * u[IDX(3,j,k)]
                            +  4.0 * u[IDX(4,j,k)]
                            -        u[IDX(5,j,k)]
                          ) * idx_by_2;

        if (betax[IDX(4,j,k)] > 0.0) {
          Dxu[IDX(4,j,k)] = ( -  3.0 * u[IDX(4,j,k)]
                              +  4.0 * u[IDX(5,j,k)]
                              -        u[IDX(6,j,k)]
                            ) * idx_by_2;
        }
        else {
          Dxu[IDX(4,j,k)] = ( -         u[IDX(3,j,k)]
                               +        u[IDX(5,j,k)]
                            ) * idx_by_2;
        }

        if (betax[IDX(5,j,k)] > 0.0 ) {
          Dxu[IDX(5,j,k)] = (-  3.0 * u[IDX(4,j,k)]
                             - 10.0 * u[IDX(5,j,k)]
                             + 18.0 * u[IDX(6,j,k)]
                             -  6.0 * u[IDX(7,j,k)]
                             +        u[IDX(8,j,k)]
                           ) * idx_by_12;
        }
        else {
          Dxu[IDX(5,j,k)] = (           u[IDX(3,j,k)]
                               -  4.0 * u[IDX(4,j,k)]
                               +  3.0 * u[IDX(5,j,k)]
                            ) * idx_by_2;
        }

      }
    }
  }

  if (bflag & (1u<<OCT_DIR_RIGHT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        if ( betax[IDX(ie-3,j,k)] < 0.0 ) {
          Dxu[IDX(ie-3,j,k)] = (  - 3.0 * u[IDX(ie-3,j,k)]
                                  + 4.0 * u[IDX(ie-2,j,k)]
                                  -       u[IDX(ie-1,j,k)]
                               ) * idx_by_2;
        }
        else {
          Dxu[IDX(ie-3,j,k)] = ( -   u[IDX(ie-6,j,k)]
                            +  6.0 * u[IDX(ie-5,j,k)]
                            - 18.0 * u[IDX(ie-4,j,k)]
                            + 10.0 * u[IDX(ie-3  ,j,k)]
                            +  3.0 * u[IDX(ie-2,j,k)]
                          ) * idx_by_12;
        }

        if (betax[IDX(ie-2,j,k)] > 0.0 ) {
          Dxu[IDX(ie-2,j,k)] = (  -  u[IDX(ie-3,j,k)]
                                  +  u[IDX(ie-1,j,k)]
                               ) * idx_by_2;
        }
        else {
          Dxu[IDX(ie-2,j,k)] = (     u[IDX(ie-4,j,k)]
                             - 4.0 * u[IDX(ie-3,j,k)]
                             + 3.0 * u[IDX(ie-2,j,k)]
                               ) * idx_by_2;
        }

        Dxu[IDX(ie-1,j,k)] = (          u[IDX(ie-3,j,k)]
                                - 4.0 * u[IDX(ie-2,j,k)]
                                + 3.0 * u[IDX(ie-1,j,k)]
                             ) * idx_by_2;

      }
    }
  }


#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Dxu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }double
  }
#endif

}


/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42adv_y(double * const  Dyu, const double * const  u,
                  const double dy, const unsigned int *sz,
                  const double * const betay, unsigned bflag)
{

  const double idy = 1.0/dy;
  const double idy_by_2 = 0.50 * idy;
  const double idy_by_12 = idy / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        if (betay[pp] > 0.0 ) {
          Dyu[pp] = ( -  3.0 * u[pp-nx]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+nx]
                      -  6.0 * u[pp+2*nx]
                      +        u[pp+3*nx]
                    ) * idy_by_12;
        }
        else {
          Dyu[pp] = ( - u[pp-3*nx]
                      +  6.0 * u[pp-2*nx]
                      - 18.0 * u[pp-nx]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+nx]
                    ) * idy_by_12;
        }

        if (betay[pp+1] > 0.0 ) {
          Dyu[pp+1] = ( -  3.0 * u[pp+1-nx]
                      - 10.0 * u[pp+1]
                      + 18.0 * u[pp+1+nx]
                      -  6.0 * u[pp+1+2*nx]
                      +        u[pp+1+3*nx]
                    ) * idy_by_12;
        }
        else {
          Dyu[pp+1] = ( - u[pp+1-3*nx]
                      +  6.0 * u[pp+1-2*nx]
                      - 18.0 * u[pp+1-nx]
                      + 10.0 * u[pp+1]
                      +  3.0 * u[pp+1+nx]
                    ) * idy_by_12;
        }


        if (betay[pp+2] > 0.0 ) {
          Dyu[pp+2] = ( -  3.0 * u[pp+2-nx]
                      - 10.0 * u[pp+2]
                      + 18.0 * u[pp+2+nx]
                      -  6.0 * u[pp+2+2*nx]
                      +        u[pp+2+3*nx]
                    ) * idy_by_12;
        }
        else {
          Dyu[pp+2] = ( - u[pp+2-3*nx]
                      +  6.0 * u[pp+2-2*nx]
                      - 18.0 * u[pp+2-nx]
                      + 10.0 * u[pp+2]
                      +  3.0 * u[pp+2+nx]
                    ) * idy_by_12;
        }
        
        if (betay[pp+3] > 0.0 ) {
          Dyu[pp+3] = ( -  3.0 * u[pp+3-nx]
                      - 10.0 * u[pp+3]
                      + 18.0 * u[pp+3+nx]
                      -  6.0 * u[pp+3+2*nx]
                      +        u[pp+3+3*nx]
                    ) * idy_by_12;
        }
        else {
          Dyu[pp+3] = ( - u[pp+3-3*nx]
                      +  6.0 * u[pp+3-2*nx]
                      - 18.0 * u[pp+3-nx]
                      + 10.0 * u[pp+3]
                      +  3.0 * u[pp+3+nx]
                    ) * idy_by_12;
        }
        
        // if (betay[pp+4] > 0.0 ) {
        //   Dyu[pp+4] = ( -  3.0 * u[pp-nx]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+nx]
        //               -  6.0 * u[pp+2*nx]
        //               +        u[pp+3*nx]
        //             ) * idy_by_12;
        // }
        // else {
        //   Dyu[pp+4] = ( - u[pp-3*nx]
        //               +  6.0 * u[pp-2*nx]
        //               - 18.0 * u[pp-nx]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+nx]
        //             ) * idy_by_12;
        // }
        
        // if (betay[pp+5] > 0.0 ) {
        //   Dyu[pp+5] = ( -  3.0 * u[pp-nx]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+nx]
        //               -  6.0 * u[pp+2*nx]
        //               +        u[pp+3*nx]
        //             ) * idy_by_12;
        // }
        // else {
        //   Dyu[pp+5] = ( - u[pp-3*nx]
        //               +  6.0 * u[pp-2*nx]
        //               - 18.0 * u[pp-nx]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+nx]
        //             ) * idy_by_12;
        // }
        
        // if (betay[pp+6] > 0.0 ) {
        //   Dyu[pp+6] = ( -  3.0 * u[pp-nx]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+nx]
        //               -  6.0 * u[pp+2*nx]
        //               +        u[pp+3*nx]
        //             ) * idy_by_12;
        // }
        // else {
        //   Dyu[pp+6] = ( - u[pp-3*nx]
        //               +  6.0 * u[pp-2*nx]
        //               - 18.0 * u[pp-nx]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+nx]
        //             ) * idy_by_12;
        // }
        
        // if (betay[pp+7] > 0.0 ) {
        //   Dyu[pp+7] = ( -  3.0 * u[pp-nx]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+nx]
        //               -  6.0 * u[pp+2*nx]
        //               +        u[pp+3*nx]
        //             ) * idy_by_12;
        // }
        // else {
        //   Dyu[pp+7] = ( - u[pp-3*nx]
        //               +  6.0 * u[pp-2*nx]
        //               - 18.0 * u[pp-nx]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+nx]
        //             ) * idy_by_12;
        // }
      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        if (betay[pp] > 0.0 ) {
          Dyu[pp] = ( -  3.0 * u[pp-nx]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+nx]
                      -  6.0 * u[pp+2*nx]
                      +        u[pp+3*nx]
                    ) * idy_by_12;
        }
        else {
          Dyu[pp] = ( - u[pp-3*nx]
                      +  6.0 * u[pp-2*nx]
                      - 18.0 * u[pp-nx]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+nx]
                    ) * idy_by_12;
        }
      }
    }
  }


  if (bflag & (1u<<OCT_DIR_DOWN)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        Dyu[IDX(i,3,k)] = ( -  3.0 * u[IDX(i,3,k)]
                            +  4.0 * u[IDX(i,4,k)]
                            -        u[IDX(i,5,k)]
                          ) * idy_by_2;

        if (betay[IDX(i,4,k)] > 0.0) {
          Dyu[IDX(i,4,k)] = ( -  3.0 * u[IDX(i,4,k)]
                              +  4.0 * u[IDX(i,5,k)]
                              -        u[IDX(i,6,k)]
                            ) * idy_by_2;
        }
        else {
          Dyu[IDX(i,4,k)] = ( -         u[IDX(i,3,k)]
                               +        u[IDX(i,5,k)]
                            ) * idy_by_2;
        }

        if (betay[IDX(i,5,k)] > 0.0 ) {
          Dyu[IDX(i,5,k)] = ( -  3.0 * u[IDX(i,4,k)]
                              - 10.0 * u[IDX(i,5,k)]
                              + 18.0 * u[IDX(i,6,k)]
                              -  6.0 * u[IDX(i,7,k)]
                             +         u[IDX(i,8,k)]
                           ) * idy_by_12;
        }
        else {
          Dyu[IDX(i,5,k)] = (           u[IDX(i,3,k)]
                               -  4.0 * u[IDX(i,4,k)]
                               +  3.0 * u[IDX(i,5,k)]
                            ) * idy_by_2;
        }
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_UP)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
        if ( betay[IDX(i,je-3,k)] < 0.0 ) {
          Dyu[IDX(i,je-3,k)] = (  - 3.0 * u[IDX(i,je-3,k)]
                                + 4.0 * u[IDX(i,je-2,k)]
                                -       u[IDX(i,je-1,k)]
                             ) * idy_by_2;
        }
        else {
          Dyu[IDX(i,je-3,k)] = ( -   u[IDX(i,je-6,k)]
                            +  6.0 * u[IDX(i,je-5,k)]
                            - 18.0 * u[IDX(i,je-4,k)]
                            + 10.0 * u[IDX(i,je-3,k)]
                            +  3.0 * u[IDX(i,je-2,k)]
                          ) * idy_by_12;
        }

        if (betay[IDX(i,je-2,k)] > 0.0 ) {
          Dyu[IDX(i,je-2,k)] = (  -  u[IDX(i,je-3,k)]
                                  +  u[IDX(i,je-1,k)]
                               ) * idy_by_2;
        }
        else {
          Dyu[IDX(i,je-2,k)] = (     u[IDX(i,je-4,k)]
                             - 4.0 * u[IDX(i,je-3,k)]
                             + 3.0 * u[IDX(i,je-2,k)]
                               ) * idy_by_2;
        }

        Dyu[IDX(i,je-1,k)] = (          u[IDX(i,je-3,k)]
                                - 4.0 * u[IDX(i,je-2,k)]
                                + 3.0 * u[IDX(i,je-1,k)]
                             ) * idy_by_2;

      }
    }
  }


#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Dyu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}


/*----------------------------------------------------------------------;
 *
 *
 *
 *----------------------------------------------------------------------*/
void deriv42adv_z(double * const  Dzu, const double * const  u,
                  const double dz, const unsigned int *sz,
                  const double * const betaz, unsigned bflag)
{

  const double idz = 1.0/dz;
  const double idz_by_2 = 0.50 * idz;
  const double idz_by_12 = idz / 12.0;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  const int n = nx * ny;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
        int pp = IDX(i,j,k);
        if (betaz[pp] > 0.0 ) {
          Dzu[pp] = ( -  3.0 * u[pp-n]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+n]
                      -  6.0 * u[pp+2*n]
                      +        u[pp+3*n]
                    ) * idz_by_12;
        }
        else {
          Dzu[pp] = ( -        u[pp-3*n]
                      +  6.0 * u[pp-2*n]
                      - 18.0 * u[pp-n]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+n]
                    ) * idz_by_12;
        }
        
        if (betaz[pp+1] > 0.0 ) {
          Dzu[pp+1] = ( -  3.0 * u[pp+1-n]
                      - 10.0 * u[pp+1]
                      + 18.0 * u[pp+1+n]
                      -  6.0 * u[pp+1+2*n]
                      +        u[pp+1+3*n]
                    ) * idz_by_12;
        }
        else {
          Dzu[pp+1] = ( -        u[pp+1-3*n]
                      +  6.0 * u[pp+1-2*n]
                      - 18.0 * u[pp+1-n]
                      + 10.0 * u[pp+1]
                      +  3.0 * u[pp+1+n]
                    ) * idz_by_12;
        }
        
        if (betaz[pp+2] > 0.0 ) {
          Dzu[pp+2] = ( -  3.0 * u[pp+2-n]
                      - 10.0 * u[pp+2]
                      + 18.0 * u[pp+2+n]
                      -  6.0 * u[pp+2+2*n]
                      +        u[pp+2+3*n]
                    ) * idz_by_12;
        }
        else {
          Dzu[pp+2] = ( -        u[pp+2-3*n]
                      +  6.0 * u[pp+2-2*n]
                      - 18.0 * u[pp+2-n]
                      + 10.0 * u[pp+2]
                      +  3.0 * u[pp+2+n]
                    ) * idz_by_12;
        }
        
        if (betaz[pp+3] > 0.0 ) {
          Dzu[pp+3] = ( -  3.0 * u[pp+3-n]
                      - 10.0 * u[pp+3]
                      + 18.0 * u[pp+3+n]
                      -  6.0 * u[pp+3+2*n]
                      +        u[pp+3+3*n]
                    ) * idz_by_12;
        }
        else {
          Dzu[pp+3] = ( -        u[pp+3-3*n]
                      +  6.0 * u[pp+3-2*n]
                      - 18.0 * u[pp+3-n]
                      + 10.0 * u[pp+3]
                      +  3.0 * u[pp+3+n]
                    ) * idz_by_12;
        }
        
        // if (betaz[pp+4] > 0.0 ) {
        //   Dzu[pp+4] = ( -  3.0 * u[pp-n]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+n]
        //               -  6.0 * u[pp+2*n]
        //               +        u[pp+3*n]
        //             ) * idz_by_12;
        // }
        // else {
        //   Dzu[pp+4] = ( -        u[pp-3*n]
        //               +  6.0 * u[pp-2*n]
        //               - 18.0 * u[pp-n]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+n]
        //             ) * idz_by_12;
        // }
        
        // if (betaz[pp+5] > 0.0 ) {
        //   Dzu[pp+5] = ( -  3.0 * u[pp-n]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+n]
        //               -  6.0 * u[pp+2*n]
        //               +        u[pp+3*n]
        //             ) * idz_by_12;
        // }
        // else {
        //   Dzu[pp+5] = ( -        u[pp-3*n]
        //               +  6.0 * u[pp-2*n]
        //               - 18.0 * u[pp-n]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+n]
        //             ) * idz_by_12;
        // }
        
        // if (betaz[pp+6] > 0.0 ) {
        //   Dzu[pp+6] = ( -  3.0 * u[pp-n]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+n]
        //               -  6.0 * u[pp+2*n]
        //               +        u[pp+3*n]
        //             ) * idz_by_12;
        // }
        // else {
        //   Dzu[pp+6] = ( -        u[pp-3*n]
        //               +  6.0 * u[pp-2*n]
        //               - 18.0 * u[pp-n]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+n]
        //             ) * idz_by_12;
        // }
        
        // if (betaz[pp+7] > 0.0 ) {
        //   Dzu[pp+7] = ( -  3.0 * u[pp-n]
        //               - 10.0 * u[pp]
        //               + 18.0 * u[pp+n]
        //               -  6.0 * u[pp+2*n]
        //               +        u[pp+3*n]
        //             ) * idz_by_12;
        // }
        // else {
        //   Dzu[pp+7] = ( -        u[pp-3*n]
        //               +  6.0 * u[pp-2*n]
        //               - 18.0 * u[pp-n]
        //               + 10.0 * u[pp]
        //               +  3.0 * u[pp+n]
        //             ) * idz_by_12;
        // }

      }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
        int pp = IDX(i,j,k);
        if (betaz[pp] > 0.0 ) {
          Dzu[pp] = ( -  3.0 * u[pp-n]
                      - 10.0 * u[pp]
                      + 18.0 * u[pp+n]
                      -  6.0 * u[pp+2*n]
                      +        u[pp+3*n]
                    ) * idz_by_12;
        }
        else {
          Dzu[pp] = ( -        u[pp-3*n]
                      +  6.0 * u[pp-2*n]
                      - 18.0 * u[pp-n]
                      + 10.0 * u[pp]
                      +  3.0 * u[pp+n]
                    ) * idz_by_12;
        }
      }
    }
  }


  if (bflag & (1u<<OCT_DIR_BACK)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        Dzu[IDX(i,j,3)] = ( -  3.0 * u[IDX(i,j,3)]
                            +  4.0 * u[IDX(i,j,4)]
                            -        u[IDX(i,j,5)]
                          ) * idz_by_2;

        if (betaz[IDX(i,j,4)] > 0.0) {
          Dzu[IDX(i,j,4)] = ( -  3.0 * u[IDX(i,j,4)]
                              +  4.0 * u[IDX(i,j,5)]
                              -        u[IDX(i,j,6)]
                            ) * idz_by_2;
        }
        else {
          Dzu[IDX(i,j,4)] = ( -         u[IDX(i,j,3)]
                               +        u[IDX(i,j,5)]
                            ) * idz_by_2;
        }

        if (betaz[IDX(i,j,5)] > 0.0 ) {
          Dzu[IDX(i,j,5)] = ( -  3.0 * u[IDX(i,j,4)]
                              - 10.0 * u[IDX(i,j,5)]
                              + 18.0 * u[IDX(i,j,6)]
                              -  6.0 * u[IDX(i,j,7)]
                             +         u[IDX(i,j,8)]
                           ) * idz_by_12;
        }
        else {
          Dzu[IDX(i,j,5)] = (           u[IDX(i,j,3)]
                               -  4.0 * u[IDX(i,j,4)]
                               +  3.0 * u[IDX(i,j,5)]
                            ) * idz_by_2;
        }
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_FRONT)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        if ( betaz[IDX(i,j,ke-3)] < 0.0 ) {
          Dzu[IDX(i,j,ke-3)] = (  - 3.0 * u[IDX(i,j,ke-3)]
                                  + 4.0 * u[IDX(i,j,ke-2)]
                                  -       u[IDX(i,j,ke-1)]
                               ) * idz_by_2;
        }
        else {
          Dzu[IDX(i,j,ke-3)] = ( -   u[IDX(i,j,ke-6)]
                            +  6.0 * u[IDX(i,j,ke-5)]
                            - 18.0 * u[IDX(i,j,ke-4)]
                            + 10.0 * u[IDX(i,j,ke-3)]
                            +  3.0 * u[IDX(i,j,ke-2)]
                          ) * idz_by_12;
        }

        if (betaz[IDX(i,j,ke-2)] > 0.0 ) {
          Dzu[IDX(i,j,ke-2)] = (  -  u[IDX(i,j,ke-3)]
                                  +  u[IDX(i,j,ke-1)]
                               ) * idz_by_2;
        }
        else {
          Dzu[IDX(i,j,ke-2)] = (     u[IDX(i,j,ke-4)]
                             - 4.0 * u[IDX(i,j,ke-3)]
                             + 3.0 * u[IDX(i,j,ke-2)]
                               ) * idz_by_2;
        }

        Dzu[IDX(i,j,ke-1)] = (          u[IDX(i,j,ke-3)]
                                - 4.0 * u[IDX(i,j,ke-2)]
                                + 3.0 * u[IDX(i,j,ke-1)]
                             ) * idz_by_2;
      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Dzu[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_x(double * const  Du, const double * const  u,
                const double dx, const unsigned int *sz, unsigned bflag)
{

  double pre_factor_6_dx = -1.0 / 64.0 / dx;

  double smr3=59.0/48.0*64*dx;
  double smr2=43.0/48.0*64*dx;
  double smr1=49.0/48.0*64*dx;
  double spr3=smr3;
  double spr2=smr2;
  double spr1=smr1;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dx *
                         (
                         -      u[pp-3]
                         +  6.0*u[pp-2]
                         - 15.0*u[pp-1]
                         + 20.0*u[pp]
                         - 15.0*u[pp+1]
                         +  6.0*u[pp+2]
                         -      u[pp+3]
                         );
          
          Du[pp+1] = pre_factor_6_dx *
                         (
                         -      u[pp-2]
                         +  6.0*u[pp-1]
                         - 15.0*u[pp]
                         + 20.0*u[pp+1]
                         - 15.0*u[pp+2]
                         +  6.0*u[pp+3]
                         -      u[pp+4]
                         );
          
          Du[pp+2] = pre_factor_6_dx *
                         (
                         -      u[pp-1]
                         +  6.0*u[pp]
                         - 15.0*u[pp+1]
                         + 20.0*u[pp+2]
                         - 15.0*u[pp+3]
                         +  6.0*u[pp+4]
                         -      u[pp+5]
                         );
          
          Du[pp+3] = pre_factor_6_dx *
                         (
                         -      u[pp]
                         +  6.0*u[pp+1]
                         - 15.0*u[pp+2]
                         + 20.0*u[pp+3]
                         - 15.0*u[pp+4]
                         +  6.0*u[pp+5]
                         -      u[pp+6]
                         );
          
          // Du[pp+4] = pre_factor_6_dx *
          //                (
          //                -      u[pp-3]
          //                +  6.0*u[pp-2]
          //                - 15.0*u[pp-1]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+1]
          //                +  6.0*u[pp+2]
          //                -      u[pp+3]
          //                );
          
          // Du[pp+5] = pre_factor_6_dx *
          //                (
          //                -      u[pp-3]
          //                +  6.0*u[pp-2]
          //                - 15.0*u[pp-1]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+1]
          //                +  6.0*u[pp+2]
          //                -      u[pp+3]
          //                );
          
          // Du[pp+6] = pre_factor_6_dx *
          //                (
          //                -      u[pp-3]
          //                +  6.0*u[pp-2]
          //                - 15.0*u[pp-1]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+1]
          //                +  6.0*u[pp+2]
          //                -      u[pp+3]
          //                );
          
          // Du[pp+7] = pre_factor_6_dx *
          //                (
          //                -      u[pp-3]
          //                +  6.0*u[pp-2]
          //                - 15.0*u[pp-1]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+1]
          //                +  6.0*u[pp+2]
          //                -      u[pp+3]
          //                );

       }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dx *
                         (
                         -      u[pp-3]
                         +  6.0*u[pp-2]
                         - 15.0*u[pp-1]
                         + 20.0*u[pp]
                         - 15.0*u[pp+1]
                         +  6.0*u[pp+2]
                         -      u[pp+3]
                         );
       }
    }
  }


  if (bflag & (1u<<OCT_DIR_LEFT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
        Du[IDX(3,j,k)] =  (      u[IDX(6,j,k)]
                           - 3.0*u[IDX(5,j,k)]
                           + 3.0*u[IDX(4,j,k)]
                           -     u[IDX(3,j,k)]
                          )/smr3;
        Du[IDX(4,j,k)] =  (
                                u[IDX(7,j,k)]
                         -  6.0*u[IDX(6,j,k)]
                         + 12.0*u[IDX(5,j,k)]
                         - 10.0*u[IDX(4,j,k)]
                         +  3.0*u[IDX(3,j,k)]
                         )/smr2;
        Du[IDX(5,j,k)] =  (
                                u[IDX(8,j,k)]
                         -  6.0*u[IDX(7,j,k)]
                         + 15.0*u[IDX(6,j,k)]
                         - 19.0*u[IDX(5,j,k)]
                         + 12.0*u[IDX(4,j,k)]
                         -  3.0*u[IDX(3,j,k)]
                         )/smr1;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_RIGHT)) {
    for (int k = kb; k < ke; k++) {
      for (int j = jb; j < je; j++) {
         Du[IDX(ie-3,j,k)] = (
                                u[IDX(ie-6,j,k)]
                          - 6.0*u[IDX(ie-5,j,k)]
                          + 15.0*u[IDX(ie-4,j,k)]
                          - 19.0*u[IDX(ie-3,j,k)]
                          + 12.0*u[IDX(ie-2,j,k)]
                          -  3.0*u[IDX(ie-1,j,k)]
                           )/spr1;

         Du[IDX(ie-2,j,k)] = (
                                 u[IDX(ie-5,j,k)]
                          -  6.0*u[IDX(ie-4,j,k)]
                          + 12.0*u[IDX(ie-3,j,k)]
                          - 10.0*u[IDX(ie-2,j,k)]
                          +  3.0*u[IDX(ie-1,j,k)]
                           )/spr2;

         Du[IDX(ie-1,j,k)] = (
                                 u[IDX(ie-4,j,k)]
                          -  3.0*u[IDX(ie-3,j,k)]
                          +  3.0*u[IDX(ie-2,j,k)]
                          -      u[IDX(ie-1,j,k)]
                           )/spr3;
      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Du[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_y(double * const  Du, const double * const  u,
                const double dy, const unsigned int *sz, unsigned bflag)
{

  double pre_factor_6_dy = -1.0 / 64.0 / dy;

  double smr3=59.0/48.0*64*dy;
  double smr2=43.0/48.0*64*dy;
  double smr1=49.0/48.0*64*dy;
  double spr3=smr3;
  double spr2=smr2;
  double spr1=smr1;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dy *
                         (
                         -      u[pp-3*nx]
                         +  6.0*u[pp-2*nx]
                         - 15.0*u[pp-nx]
                         + 20.0*u[pp]
                         - 15.0*u[pp+nx]
                         +  6.0*u[pp+2*nx]
                         -      u[pp+3*nx]
                         );
          
          Du[pp+1] = pre_factor_6_dy *
                         (
                         -      u[pp+1-3*nx]
                         +  6.0*u[pp+1-2*nx]
                         - 15.0*u[pp+1-nx]
                         + 20.0*u[pp+1]
                         - 15.0*u[pp+1+nx]
                         +  6.0*u[pp+1+2*nx]
                         -      u[pp+1+3*nx]
                         );
          
          Du[pp+2] = pre_factor_6_dy *
                         (
                         -      u[pp+2-3*nx]
                         +  6.0*u[pp+2-2*nx]
                         - 15.0*u[pp+2-nx]
                         + 20.0*u[pp+2]
                         - 15.0*u[pp+2+nx]
                         +  6.0*u[pp+2+2*nx]
                         -      u[pp+2+3*nx]
                         );
          
          Du[pp+3] = pre_factor_6_dy *
                         (
                         -      u[pp+3-3*nx]
                         +  6.0*u[pp+3-2*nx]
                         - 15.0*u[pp+3-nx]
                         + 20.0*u[pp+3]
                         - 15.0*u[pp+3+nx]
                         +  6.0*u[pp+3+2*nx]
                         -      u[pp+3+3*nx]
                         );
          
          // Du[pp+4] = pre_factor_6_dy *
          //                (
          //                -      u[pp-3*nx]
          //                +  6.0*u[pp-2*nx]
          //                - 15.0*u[pp-nx]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+nx]
          //                +  6.0*u[pp+2*nx]
          //                -      u[pp+3*nx]
          //                );
          
          // Du[pp+5] = pre_factor_6_dy *
          //                (
          //                -      u[pp-3*nx]
          //                +  6.0*u[pp-2*nx]
          //                - 15.0*u[pp-nx]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+nx]
          //                +  6.0*u[pp+2*nx]
          //                -      u[pp+3*nx]
          //                );
          
          // Du[pp+6] = pre_factor_6_dy *
          //                (
          //                -      u[pp-3*nx]
          //                +  6.0*u[pp-2*nx]
          //                - 15.0*u[pp-nx]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+nx]
          //                +  6.0*u[pp+2*nx]
          //                -      u[pp+3*nx]
          //                );
          
          // Du[pp+7] = pre_factor_6_dy *
          //                (
          //                -      u[pp-3*nx]
          //                +  6.0*u[pp-2*nx]
          //                - 15.0*u[pp-nx]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+nx]
          //                +  6.0*u[pp+2*nx]
          //                -      u[pp+3*nx]
          //                );

       }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dy *
                         (
                         -      u[pp-3*nx]
                         +  6.0*u[pp-2*nx]
                         - 15.0*u[pp-nx]
                         + 20.0*u[pp]
                         - 15.0*u[pp+nx]
                         +  6.0*u[pp+2*nx]
                         -      u[pp+3*nx]
                         );
       }
    }
  }


  if (bflag & (1u<<OCT_DIR_DOWN)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
       Du[IDX(i,3,k)] =  (      u[IDX(i,6,k)]
                          - 3.0*u[IDX(i,5,k)]
                          + 3.0*u[IDX(i,4,k)]
                          -     u[IDX(i,3,k)]
                         )/smr3;
       Du[IDX(i,4,k)] =  (
                                u[IDX(i,7,k)]
                         -  6.0*u[IDX(i,6,k)]
                         + 12.0*u[IDX(i,5,k)]
                         - 10.0*u[IDX(i,4,k)]
                         +  3.0*u[IDX(i,3,k)]
                         )/smr2;
       Du[IDX(i,5,k)] =  (
                                u[IDX(i,8,k)]
                         -  6.0*u[IDX(i,7,k)]
                         + 15.0*u[IDX(i,6,k)]
                         - 19.0*u[IDX(i,5,k)]
                         + 12.0*u[IDX(i,4,k)]
                         -  3.0*u[IDX(i,3,k)]
                         )/smr1;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_UP)) {
    for (int k = kb; k < ke; k++) {
      for (int i = ib; i < ie; i++) {
         Du[IDX(i,je-3,k)] = (
                                 u[IDX(i,je-6,k)]
                          -  6.0*u[IDX(i,je-5,k)]
                          + 15.0*u[IDX(i,je-4,k)]
                          - 19.0*u[IDX(i,je-3,k)]
                          + 12.0*u[IDX(i,je-2,k)]
                          -  3.0*u[IDX(i,je-1,k)]
                           )/spr1;

       Du[IDX(i,je-2,k)] = (
                                 u[IDX(i,je-5,k)]
                          -  6.0*u[IDX(i,je-4,k)]
                          + 12.0*u[IDX(i,je-3,k)]
                          - 10.0*u[IDX(i,je-2,k)]
                          +  3.0*u[IDX(i,je-1,k)]
                           )/spr2;

       Du[IDX(i,je-1,k)] = (
                                 u[IDX(i,je-4,k)]
                          -  3.0*u[IDX(i,je-3,k)]
                          +  3.0*u[IDX(i,je-2,k)]
                          -      u[IDX(i,je-1,k)]
                           )/spr3;

      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Du[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}

/*----------------------------------------------------------------------
 *
 *
 *
 *----------------------------------------------------------------------*/
void ko_deriv42_z(double * const  Du, const double * const  u,
                const double dz, const unsigned *sz, unsigned bflag)
{

  double pre_factor_6_dz = -1.0 / 64.0 / dz;

  double smr3=59.0/48.0*64*dz;
  double smr2=43.0/48.0*64*dz;
  double smr1=49.0/48.0*64*dz;
  double spr3=smr3;
  double spr2=smr2;
  double spr1=smr1;

  const int nx = sz[0];
  const int ny = sz[1];
  const int nz = sz[2];
  const int ib = 3;
  const int jb = 3;
  const int kb = 3;
  const int ie = sz[0] - 3;
  const int je = sz[1] - 3;
  const int ke = sz[2] - 3;

  const int n = nx * ny;

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i+=4) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dz *
                         (
                         -      u[pp-3*n]
                         +  6.0*u[pp-2*n]
                         - 15.0*u[pp-n]
                         + 20.0*u[pp]
                         - 15.0*u[pp+n]
                         +  6.0*u[pp+2*n]
                         -      u[pp+3*n]
                         );
          
          Du[pp+1] = pre_factor_6_dz *
                         (
                         -      u[pp+1-3*n]
                         +  6.0*u[pp+1-2*n]
                         - 15.0*u[pp+1-n]
                         + 20.0*u[pp+1]
                         - 15.0*u[pp+1+n]
                         +  6.0*u[pp+1+2*n]
                         -      u[pp+1+3*n]
                         );
          
          Du[pp+2] = pre_factor_6_dz *
                         (
                         -      u[pp+2-3*n]
                         +  6.0*u[pp+2-2*n]
                         - 15.0*u[pp+2-n]
                         + 20.0*u[pp+2]
                         - 15.0*u[pp+2+n]
                         +  6.0*u[pp+2+2*n]
                         -      u[pp+2+3*n]
                         );
          
          Du[pp+3] = pre_factor_6_dz *
                         (
                         -      u[pp+3-3*n]
                         +  6.0*u[pp+3-2*n]
                         - 15.0*u[pp+3-n]
                         + 20.0*u[pp+3]
                         - 15.0*u[pp+3+n]
                         +  6.0*u[pp+3+2*n]
                         -      u[pp+3+3*n]
                         );
          
          // Du[pp+4] = pre_factor_6_dz *
          //                (
          //                -      u[pp-3*n]
          //                +  6.0*u[pp-2*n]
          //                - 15.0*u[pp-n]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+n]
          //                +  6.0*u[pp+2*n]
          //                -      u[pp+3*n]
          //                );
          
          // Du[pp+5] = pre_factor_6_dz *
          //                (
          //                -      u[pp-3*n]
          //                +  6.0*u[pp-2*n]
          //                - 15.0*u[pp-n]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+n]
          //                +  6.0*u[pp+2*n]
          //                -      u[pp+3*n]
          //                );
          
          // Du[pp+6] = pre_factor_6_dz *
          //                (
          //                -      u[pp-3*n]
          //                +  6.0*u[pp-2*n]
          //                - 15.0*u[pp-n]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+n]
          //                +  6.0*u[pp+2*n]
          //                -      u[pp+3*n]
          //                );
          
          // Du[pp+7] = pre_factor_6_dz *
          //                (
          //                -      u[pp-3*n]
          //                +  6.0*u[pp-2*n]
          //                - 15.0*u[pp-n]
          //                + 20.0*u[pp]
          //                - 15.0*u[pp+n]
          //                +  6.0*u[pp+2*n]
          //                -      u[pp+3*n]
          //                );

       }
    }
  }

  #pragma omp parallel for collapse(3) schedule(static)
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = 4*(ie/4); i < ie; i++) {
          int pp = IDX(i,j,k);
          Du[pp] = pre_factor_6_dz *
                         (
                         -      u[pp-3*n]
                         +  6.0*u[pp-2*n]
                         - 15.0*u[pp-n]
                         + 20.0*u[pp]
                         - 15.0*u[pp+n]
                         +  6.0*u[pp+2*n]
                         -      u[pp+3*n]
                         );
       }
    }
  }


  if (bflag & (1u<<OCT_DIR_BACK)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        Du[IDX(i,j,3)] =  (      u[IDX(i,j,6)]
                          - 3.0*u[IDX(i,j,5)]
                          + 3.0*u[IDX(i,j,4)]
                          -     u[IDX(i,j,3)]
                         )/smr3;
        Du[IDX(i,j,4)] =  (
                                u[IDX(i,j,7)]
                         -  6.0*u[IDX(i,j,6)]
                         + 12.0*u[IDX(i,j,5)]
                         - 10.0*u[IDX(i,j,4)]
                         +  3.0*u[IDX(i,j,3)]
                         )/smr2;
        Du[IDX(i,j,5)] =  (
                                u[IDX(i,j,8)]
                         -  6.0*u[IDX(i,j,7)]
                         + 15.0*u[IDX(i,j,6)]
                         - 19.0*u[IDX(i,j,5)]
                         + 12.0*u[IDX(i,j,4)]
                         -  3.0*u[IDX(i,j,3)]
                         )/smr1;
      }
    }
  }

  if (bflag & (1u<<OCT_DIR_FRONT)) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
         Du[IDX(i,j,ke-3)] = (
                                 u[IDX(i,j,ke-6)]
                          -  6.0*u[IDX(i,j,ke-5)]
                          + 15.0*u[IDX(i,j,ke-4)]
                          - 19.0*u[IDX(i,j,ke-3)]
                          + 12.0*u[IDX(i,j,ke-2)]
                          -  3.0*u[IDX(i,j,ke-1)]
                           )/spr1;

         Du[IDX(i,j,ke-2)] = (
                                 u[IDX(i,j,ke-5)]
                          -  6.0*u[IDX(i,j,ke-4)]
                          + 12.0*u[IDX(i,j,ke-3)]
                          - 10.0*u[IDX(i,j,ke-2)]
                          +  3.0*u[IDX(i,j,ke-1)]
                           )/spr2;

         Du[IDX(i,j,ke-1)] = (
                                 u[IDX(i,j,ke-4)]
                          -  3.0*u[IDX(i,j,ke-3)]
                          +  3.0*u[IDX(i,j,ke-2)]
                          -      u[IDX(i,j,ke-1)]
                           )/spr3;
      }
    }
  }

#ifdef DEBUG_DERIVS_COMP
  for (int k = kb; k < ke; k++) {
    for (int j = jb; j < je; j++) {
      for (int i = ib; i < ie; i++) {
        int pp = IDX(i,j,k);
        if(std::isnan(Du[pp])) std::cout<<"NAN detected function "<<__func__<<" file: "<<__FILE__<<" line: "<<__LINE__<<std::endl;
      }
    }
  }
#endif

}


void cpy_unzip_padd(double * const  Du, const double * const  u,const unsigned int *sz, unsigned bflag)
{
    const int nx = sz[0];
    const int ny = sz[1];
    const int nz = sz[2];

    for(unsigned int k=0;k<sz[2];k++)
      for(unsigned int j=0;j<sz[1];j++)
        for(unsigned int i=0;i<sz[0];i++)
            if((i<3||i>=sz[0]-3) || (j<3||j>=sz[0]-3)|| (k<3||k>=sz[2]-3))
                Du[IDX(i,j,k)]=u[IDX(i,j,k)];


}
