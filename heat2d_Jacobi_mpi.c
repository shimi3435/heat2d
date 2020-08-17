#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<mpi.h>

int main(int argc, char **argv){
    const double l = 1.0;
    const int N = 250; 
    const double kappa = 1.0;
    const double half = 1/2.0;
    const double quart = 1/4.0;
    const double rec_N = 1/(double)N;
    const double coeff_q = -1*rec_N*rec_N/kappa;
    const double delta_xy = l*rec_N;
    const double eps = 1.0e-5;

    const int tag1 = 100;
    const int tag2 = 200;
    MPI_Status status;

    int i, j, count, nprocs, myrank;
    double err, err_temp, err_local;
    double tstart, tend;
    double t_old[(N+1)*(N+1)];
    double t_new[(N+1)*(N+1)];
    double t_reduce[(N+1)*(N+1)];
    double diag[(N+1)*(N+1)];
    double rec_diag[(N+1)*(N+1)];
    double upper1[(N+1)*(N+1)];
    double upper2[(N+1)*(N+1)];
    double lower1[(N+1)*(N+1)];
    double lower2[(N+1)*(N+1)];
    double q[(N+1)*(N+1)];

    double x_place;
    double y_place;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int istart = (N+1)*myrank/nprocs + 1;
    int iend = myrank == nprocs-1 ? (N+1)*(myrank+1)/nprocs - 2 : (N+1)*(myrank+1)/nprocs;

    int upper_proc = myrank != 0 ? myrank - 1 : MPI_PROC_NULL;
    int lower_proc = myrank != nprocs-1 ? myrank + 1 : MPI_PROC_NULL;

    if(myrank == 0){
        printf("threads = %d\n",omp_get_max_threads());
        printf("nprocs = %d\n",nprocs);
    }
    
    for(i=0; i<nprocs; i++){
        if(myrank == i){
            printf("myrank = %d istart = %d, iend = %d, upper_proc = %d, lower_proc = %d\n", myrank, istart, iend, upper_proc, lower_proc);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    #pragma omp parallel for private(i,j,y_place,x_place)
    for(i=0; i<=N; i++){
        y_place = i*delta_xy;
        for(j=0; j<=N; j++){
            x_place = j*delta_xy;
            q[i*(N+1) + j] = (x_place - half)*(x_place - half) + (y_place - half)*(y_place - half) <= quart*quart ? coeff_q*60.0 : 0;
            t_old[i*(N+1) + j] = 0;
            t_new[i*(N+1) + j] = 0;
            t_reduce[i*(N+1) + j] = 0;
            diag[i*(N+1) + j] = 0;
            rec_diag[i*(N+1) + j] = 0;
            upper1[i*(N+1) + j] = 0;
            upper2[i*(N+1) + j] = 0;
            lower1[i*(N+1) + j] = 0;
            lower2[i*(N+1) + j] = 0;
        }
    }

    #pragma omp parallel for private(i,j)
    for(i=1;i<N;i++){
        for(j=1;j<N;j++){
            diag[i*(N+1) + j] = -4;
            rec_diag[i*(N+1) + j] = 1/diag[i*(N+1) + j];
            if(i!=1){
                lower2[i*(N+1) + j] = 1;
            }
            if(i!=N-1){
                upper2[i*(N+1) + j] = 1;
            }
            if(j!=1){
                lower1[i*(N+1) + j] = 1;
            }
            if(j!=N-1){
                upper1[i*(N+1) + j] = 1;
            }
        }
    }

    err = 0;
    #pragma omp parallel for private(i,j,err_temp) reduction(+:err)
    for(i=1; i<N; i++){
        for(j=1; j<N; j++){
            err_temp = upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_old[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_old[i*(N+1) + j-1] + diag[i*(N+1) + j]*t_old[i*(N+1) + j] - q[i*(N+1) + j];
            err += err_temp*err_temp;
        }
    }
    err = sqrt(err);

    MPI_Barrier(MPI_COMM_WORLD);
    tstart = MPI_Wtime();

    #pragma omp parallel
    {

    count = 0;
    while(err > eps){
        count++;

        #pragma omp for private(i,j)
        for(i=istart; i<=iend; i++){
            for(j=1; j<N; j++){
                t_old[i*(N+1) + j] = t_new[i*(N+1) + j];
            }
        }

        #pragma omp single
        {
            MPI_Sendrecv(&t_old[iend*(N+1)], N+1, MPI_DOUBLE, lower_proc, tag1, &t_old[(istart-1)*(N+1)], N+1, MPI_DOUBLE, upper_proc, tag1, MPI_COMM_WORLD, &status);
            MPI_Sendrecv(&t_old[istart*(N+1)], N+1, MPI_DOUBLE, upper_proc, tag2, &t_old[(iend+1)*(N+1)], N+1, MPI_DOUBLE, lower_proc, tag2, MPI_COMM_WORLD, &status);
        }

        #pragma omp for private(i,j)
        for(i=istart; i<=iend; i++){
            for(j=1; j<N; j++){
                t_new[i*(N+1) + j] = rec_diag[i*(N+1) +j]*(q[i*(N+1) + j] - (upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_old[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_old[i*(N+1) + j-1]));
            }
        }

        err = 0;
        err_local = 0;

        #pragma omp single
        {
            MPI_Sendrecv(&t_new[iend*(N+1)], N+1, MPI_DOUBLE, lower_proc, tag1, &t_new[(istart-1)*(N+1)], N+1, MPI_DOUBLE, upper_proc, tag1, MPI_COMM_WORLD, &status);
            MPI_Sendrecv(&t_new[istart*(N+1)], N+1, MPI_DOUBLE, upper_proc, tag2, &t_new[(iend+1)*(N+1)], N+1, MPI_DOUBLE, lower_proc, tag2, MPI_COMM_WORLD, &status);
        }

        #pragma omp for private(i,j,err_temp) reduction(+:err_local)
        for(i=istart; i<=iend; i++){
            for(j=1; j<N; j++){
                err_temp = upper2[i*(N+1) + j]*t_new[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_new[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1] + diag[i*(N+1) + j]*t_new[i*(N+1) + j] - q[i*(N+1) + j];
                err_local += err_temp*err_temp;
            }
        }

        #pragma omp single
        {
            MPI_Allreduce(&err_local, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        #pragma omp single
        {
            err = sqrt(err);
        }   
    }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tend = MPI_Wtime();

    for(j=0; j<=N; j++){
        t_new[(istart-1)*(N+1) + j] = 0;
        t_new[(iend+1)*(N+1) + j] = 0;
    }

    MPI_Reduce(&t_new[0], &t_reduce[0], (N+1)*(N+1), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
    if(myrank == 0){
        if(omp_get_max_threads()==1) printf("count = %d\n",count);
        printf(" err = %22.10f\n", err);
        printf("time = %22.10f\n",tend-tstart);
        FILE *file;

        file = fopen("heat2d_Jacobi_mpi.dat", "w");
        if(file == NULL){
            printf("cannot open file\n");
            return -1;
        }

        for(i=0; i<=N; i++){
            y_place = i*delta_xy;
            for(j=0; j<=N; j++){
                x_place = j*delta_xy;
                fprintf(file, "%lf %lf %lf\n",x_place, y_place, t_reduce[i*(N+1) + j]);
            }
        }

        printf("result > heat2d_Jacobi_mpi.dat\n");
    }

    MPI_Finalize();

    return 0;
}