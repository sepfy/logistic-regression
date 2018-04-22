#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int S;
int Xn;
int Yn;
int m =2;

double **xx, **y;

void load() {


    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char *del = ",";
    char *pch;
    int in = 0;
    int nn =0;
    fp = fopen("train.csv", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
     int i=0, j=0;
     getline(&line, &len, fp);
     pch= strtok(line, del);
     while(pch != NULL) {

       //printf("%s\n", s);
       pch = strtok(NULL, del);
       in++;
       nn++;
     }
     in = in - 1;
     nn = nn-1;
    int mm = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
      mm++;
        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", line);
    }
    //printf("%d\n", mm);
    fclose(fp);
    int N = 10;
    xx = (double**)malloc(nn*sizeof(double*));
    y = (double**)malloc(N*sizeof(double*));
     
    for(i = 0; i < nn; i++) {
      xx[i] = (double*)malloc(mm*sizeof(double));
    } 
    for(i = 0; i < N; i++) {
     y[i] = (double*)malloc(mm*sizeof(double));
    } 
    fp = fopen("train.csv", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
     getline(&line, &len, fp);
     //printf("%d\n", in);
     j=0;
     while ((read = getline(&line, &len, fp)) != -1) {

       i=0;
       pch = strtok(line, del);
       y[atoi(pch)][j] = 1.0;
         pch = strtok(NULL, del);
       while(pch != NULL) {
         if(atof(pch) > 0) 
           xx[i][j] = 1.0;
         else
           xx[i][j] = 0.0;

         //printf("%d, %g\n", i, xx[i][j]);
         pch = strtok(NULL, del);
         i++;
       }
        j++;
    }
       
    for(i=0;i<nn;i++) {
      if(i%28==0)
        printf("\n");
        printf("%.0f",xx[i][0]);
      //printf("%g ", xx[i][0]);
    }
    for(i=0;i<N;i++) 
      printf("\n %g ", y[i][0]);
    printf("\n");
    fclose(fp);
    if (line)
        free(line);
    S = mm;
    Xn = nn;
    Yn = 10;
}

void init(int *N, double ***W, double ***J, double **H, double **D) {

    printf("test1 %d\n", N[1]);
}


void a_s(int m, int n, double **W, double *x, double *z) {
  int i ,j;
  for(i = 0; i < m; i++) {
      z[i] = 0;
      for(j = 0; j < n; j++) {
          z[i] += W[i][j]*x[j];
    }
  }

    //    printf("%.2f ", max);
    //printf("\n");
  for(i = 0; i < m; i++) {
      z[i] = 1/(1.0 + exp(-1.0*z[i]));

  }

}

void a(int m, int n, double **W, double *x, double *z) {

    int i, j;
    double tmp = 0;
    double max = 0;
    for(i = 0; i < m; i++) {
        z[i] = 0;
        for(j = 0; j < n; j++) {
            z[i] += W[i][j]*x[j];
        }

        if(z[i] > max)
            max = z[i];
    }

    //    printf("%.2f ", max);
    //printf("\n");
    for(i = 0; i < m; i++) {
        z[i] = exp(z[i] - max);
        tmp += z[i];
    }
    for(i = 0; i < m; i++)  {
        z[i] /= tmp;
        //printf("%.2f ", y[i]);
    }
    //printf("\n");
}

void f(int *N, double ***W, double **H) {

    for(int l = 0; l < m; l++) {
        if(l==(m-1))
          a(N[l+1], N[l], W[l], H[l], H[l+1]);
        else
          a_s(N[l+1], N[l], W[l], H[l], H[l+1]);
         
    }
}

double c(int *N, double **H, int ss) {
    int i;

    double tmp = 0;
    double tmp1 = 0;
    for(i = 0; i < N[m]; i++) {
        tmp -= y[i][ss]*log(H[m][i]) + (1 - y[i][ss])*log(1 - H[m][i]);
        //printf("%g ", tmp1);
        //printf("%g ", y[i][0]);
        //printf("%g ", H[m][i]);
        //printf("\n");
    }
    return tmp;
}

void b(int *N, double ***W, double ***J, double **H, double **D, int ss) {

    int i, j, k, l;
    for(l = m-1; l >= 0; l--) {
        for(i = 0; i < N[l+1]; i++) {
            if(l == (m-1)) {
                //D[l][i] = -(y[i][ss]*(1 - H[l+1][i]) - (1 - y[i][ss])*H[l+1][i]);
                D[l][i] = -(1/(double)S)*(y[i][ss] - H[l+1][i]);
            }
            else {
                D[l][i] = 0;
                for(k = 0; k < N[l+2]; k++) 
                    D[l][i] += D[l+1][k]*W[l+1][k][i]*(1 - H[l+1][i])*H[l+1][i];
            }

            for(j = 0; j < N[l]; j++) {
                //J[l][i][j] += D[l][i]*H[l][j];
                W[l][i][j] -= 1*D[l][i]*H[l][j];
            }
        }
    }

}


int main(void) {

    //double **x, **y;


    int i, j, k,l;
    double ***W, ***J;
    double **H, **D;
    int *N;

    load();
    printf("(%d, %d, %d)\n", S, Xn, Yn);

    //allocate
    N = (int*)malloc((m+1)*sizeof(int));
    for(i = 1; i < m; i++)
      N[i] = 10;
    N[0] = Xn;
    N[m] = Yn;

    W = (double***)malloc(m*sizeof(double**));
    J = (double***)malloc(m*sizeof(double**));
  
    for(i = 0; i < m; i++) {
        W[i] = (double**)malloc(N[i+1]*sizeof(double*));
        J[i] = (double**)malloc(N[i+1]*sizeof(double*));
    }

    for(i = 0; i < m; i++) {
        for(j = 0; j < N[i+1]; j++) {
            W[i][j] = (double*)malloc(N[i]*sizeof(double));
            J[i][j] = (double*)malloc(N[i]*sizeof(double));
        }
    }

    H = (double**)malloc((m+1)*sizeof(double*));
    D = (double**)malloc((m+1)*sizeof(double*));
    for(i = 0; i < m+1; i++){ 
      H[i] = (double*)malloc(N[i]*sizeof(double));
      D[i] = (double*)malloc(N[i]*sizeof(double));
    }
   
    for(l = 0; l < m; l++) 
        for(i = 0; i < N[l+1]; i++) 
            for(j = 0; j < N[l]; j++) {
                W[l][i][j] = 1.0; 
                J[l][i][j] = 0.0;
            }

    int iter, iter_max = 1000;
    double ce = 0;
    int ss = 0;
    S = 2;
    int s, p,q;
    double alpha = 1;
    printf("S=%g\n", (1/(double)S));



    for(iter = 0; iter < iter_max; iter++) {
        ce = 0;
        for(i = 0; i < S; i++) {

            for(j = 0; j < N[0]; j++)
                H[0][j] = xx[j][i];   


            f(N, W, H);
            ce += (1/(double)S)*c(N, H, i);
            b(N, W, J, H, D, i);

        }
  //  printf("iter: \n", iter); 
  //  for(p = 0; p < N[2]; p++) {
  //    for(q = 0; q < N[1]; q++) {
   //     printf("%.1f ", W[1][p][q]);
   //   }
   //   printf("\n");
   // }
          //  for(l = 0; l < m; l++) 
          //      for(p = 0; p < N[l+1]; p++) 
          //          for(q = 0; q < N[l]; q++) {
          //              W[l][p][q] -= (1/(double)S)*alpha*J[l][p][q]; 
          //             J[l][p][q] = 0; 
          //         }
        if(iter % 1000 ==0)
            printf("ce = %g\n", ce);
        if(ce < 1.0e-1)
            break;

    }


    double max;
    int idx;
    double err = 0;
    for(s = 0; s < S; s++) {

        for(i = 0; i < N[0]; i++)
            H[0][i] = xx[i][s];   

        f(N, W, H);
//    printf("\n");
//    printf("\n");
//    for(i = 0; i < N[1]; i++) {
//      printf("%g", H[1][i]);
//    }
        max = 0;
        for(i = 0; i < N[m]; i++) { 
            if(H[m][i] > max) {
                max = H[m][i];
                idx = i;
            }
        }
        if(y[idx][s] != 1) 
            err += 1.0;
    }

    printf("error = %g\n", err/(double)S);
    return 0;
}

