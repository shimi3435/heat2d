#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

int main(void){
    const double l = 1.0;
    const int N = 250; 
    const double kappa = 1.0;
    const double half = 1/2.0;
    const double quart = 1/4.0;
    const double rec_N = 1/(double)N;
    const double coeff_q = -1*rec_N*rec_N/kappa;
    const double delta_xy = l*rec_N;
    const double eps = 1.0e-5;

    int i, j, count;
    double err, err_temp;
    double tstart, tend;
    double t_old[(N+1)*(N+1)];
    double t_new[(N+1)*(N+1)];
    double diag[(N+1)*(N+1)];
    double rec_diag[(N+1)*(N+1)];
    double upper1[(N+1)*(N+1)];
    double upper2[(N+1)*(N+1)];
    double lower1[(N+1)*(N+1)];
    double lower2[(N+1)*(N+1)];
    double q[(N+1)*(N+1)];

    double x_place;
    double y_place;

    #pragma omp parallel for private(i,j,y_place,x_place)
    for(i=0; i<=N; i++){
        y_place = i*delta_xy;
        for(j=0; j<=N; j++){
            x_place = j*delta_xy;
            q[i*(N+1) + j] = (x_place - half)*(x_place - half) + (y_place - half)*(y_place - half) <= quart*quart ? coeff_q*60.0 : 0;
            t_old[i*(N+1) + j] = 0;
            t_new[i*(N+1) + j] = 0;
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

    tstart = omp_get_wtime();

    #pragma omp parallel
    {

    count = 0;
    while(err > eps){
        count++;

        #pragma omp for private(i,j)
        for(i=1; i<N; i++){
            for(j=1; j<N; j++){
                t_old[i*(N+1) + j] = t_new[i*(N+1) + j];
            }
        }

        //red
        #pragma omp for private(i,j)
        for(i=1; i<N; i++){
            if(i%2 == 0){
                for(j=1; j<N; j+=2){
                   t_new[i*(N+1) + j] = rec_diag[i*(N+1) +j]*(q[i*(N+1) + j] - (upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1]));
                }
            }else{
                for(j=2; j<N; j+=2){
                   t_new[i*(N+1) + j] = rec_diag[i*(N+1) +j]*(q[i*(N+1) + j] - (upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1]));
                }
            }
        }

        //black
        #pragma omp for private(i,j)
        for(i=1; i<N; i++){
            if(i%2 == 0){
                for(j=2; j<N; j+=2){
                   t_new[i*(N+1) + j] = rec_diag[i*(N+1) +j]*(q[i*(N+1) + j] - (upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1]));
                }
            }else{
                for(j=1; j<N; j+=2){
                   t_new[i*(N+1) + j] = rec_diag[i*(N+1) +j]*(q[i*(N+1) + j] - (upper2[i*(N+1) + j]*t_old[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_old[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1]));
                }
            }
        }

        err = 0;
        #pragma omp for private(i,j,err_temp) reduction(+:err)
        for(i=1; i<N; i++){
            for(j=1; j<N; j++){
                err_temp = upper2[i*(N+1) + j]*t_new[(i+1)*(N+1) + j] + lower2[i*(N+1) + j]*t_new[(i-1)*(N+1) + j] + upper1[i*(N+1) + j]*t_new[i*(N+1) + j+1] + lower1[i*(N+1) + j]*t_new[i*(N+1) + j-1] + diag[i*(N+1) + j]*t_new[i*(N+1) + j] - q[i*(N+1) + j];
                err += err_temp*err_temp;
            }
        }

        #pragma omp single
        {
        err = sqrt(err);
        }   
        //printf("count = %d\n err = %22.10f\n",count, err);
    }
    }

    tend = omp_get_wtime();

    printf("threads = %d\n",omp_get_max_threads());
    if(omp_get_max_threads()==1) printf("count = %d\n",count);
    printf(" err = %22.10f\n", err);
    printf("time = %22.10f\n",tend-tstart);

    FILE *file;

    file = fopen("heat2d_GS_RB_omp.dat", "w");
    if(file == NULL){
        printf("cannot open file\n");
        return -1;
    }

    for(i=0; i<=N; i++){
        y_place = i*delta_xy;
        for(j=0; j<=N; j++){
            x_place = j*delta_xy;
            fprintf(file, "%lf %lf %lf\n",x_place, y_place, t_new[i*(N+1) + j]);
            //fprintf(file, "%d %d %lf\n",i, j, t_new[i*(N+1) + j]);
        }
    }

    printf("result > heat2d_GS_RB_omp.dat\n");

    return 0;
}