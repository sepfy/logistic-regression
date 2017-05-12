#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

int n_features = 28*28;
int n_samples = 40000;
int n_cls = 10;
double **train_mat;
double **new_wvec, **old_wvec;
double **yvec, **fvec;



void checkSamplesCount(char *filename) {
  int count = 0;
  fstream file;
  file.open(filename);
  string line;
  while (getline(file, line,'\n')) {
    count++;
  }
  file.close();
  //n_samples = count;
  n_samples = 10000;
}


void output_train_result(char *filename) {

  fstream file;
  file.open(filename, ios::out);
  if(!file){
    cout<<"Fail to open file: "<< filename << endl;
    exit(0);
  }

  for(int i = 0; i < n_cls; i++) {
    for(int j = 0; j < n_features; j++) {
      file << new_wvec[j][i] <<",";
    }
    file << "\n";
  }

  file.close();
}


void load_data(char *filename) {

  int row = 0, col = 0;
  fstream file;
  file.open(filename);
  string line;

  getline(file, line,'\n');

  while (getline(file, line,'\n')) {
    istringstream templine(line); 
    string data;
    col = 0;
    getline(templine, data,',');

    for(int i = 0; i < n_cls; i++) {
      if(atoi(data.c_str()) == i)
        yvec[row][i] = 1;
      else
        yvec[row][i] = 0;   
    }

 
    while (getline(templine, data,',')) {
      if(atof(data.c_str()) > 0)
        train_mat[row][col] = 1;
      else
        train_mat[row][col] = 0;
      col++;
    }

    row++;
    if(row >= n_samples) break;
  }
  file.close();

}


int allocate() {
  
  train_mat = (double**)malloc(sizeof(double*)*n_samples);

  yvec = (double**)malloc(sizeof(double*)*n_samples);
  fvec = (double**)malloc(sizeof(double*)*n_samples);

  new_wvec = (double**)malloc(sizeof(double*)*n_features);
  old_wvec = (double**)malloc(sizeof(double*)*n_features);

  for(int i = 0; i < n_samples; i++) {
    train_mat[i] = (double*)malloc(sizeof(double)*n_features);
    yvec[i] = (double*)malloc(sizeof(double)*n_cls);
    fvec[i] = (double*)malloc(sizeof(double)*n_cls);
  }


  for(int i = 0; i < n_features; i++) {
    new_wvec[i] = (double*)malloc(sizeof(double)*n_cls);
    old_wvec[i] = (double*)malloc(sizeof(double)*n_cls);
  }


  for(int i = 0; i < n_features; i++) {
    for(int j = 0; j < n_cls; j++) {
      new_wvec[i][j] = 1.0;
      old_wvec[i][j] = 1.0;
    }
  }

}




void updatefvec() {

  int i, j, k;
  double tmp[10];
  double tmp1 = 0;


  for(i = 0; i < n_samples; i++) {

    tmp1 = 0;
    for(k =0; k < n_cls; k++) {

      tmp[k] = 0;

      for(j = 0; j < n_features; j++) {
        tmp[k] += train_mat[i][j]*old_wvec[j][k];
      }
      tmp[k] = exp(tmp[k]);
      tmp1 += tmp[k];
    }

    for(k =0; k < n_cls; k++)
      fvec[i][k] = tmp[k]/tmp1;

  }


}


void updatewvec(){

  for(int i = 0; i < n_features; i++) 
    for(int j = 0; j < n_cls; j++) 
      old_wvec[i][j] = new_wvec[i][j];
    
  
}




void GradientDescent(double alpha, double lambda, int max_iter, double max_err) {

  double tmp, tmp1, rel_err;
  int iter = 0, i, j, k;

  while(iter < max_iter) {
    
    updatewvec();
    updatefvec();

    for(k = 0; k < n_cls; k++) {
      for(i = 0; i < n_features; i++) {

        tmp = 0;
        for(j = 0; j < n_samples; j++) {
          tmp += (yvec[j][k] - fvec[j][k])*train_mat[j][i];
        }
        new_wvec[i][k] = old_wvec[i][k] - (alpha*(-1/n_samples)*tmp + lambda*old_wvec[i][k]);
      }
    }




    tmp  = 0;
    tmp1 = 0;
    for(k = 0; k < n_cls; k++) {
      for(i = 0; i < n_features; i++) {
        tmp  += pow(new_wvec[i][k] - old_wvec[i][k], 2);
        tmp1 += pow(new_wvec[i][k], 2);
      }
    } 

    rel_err = pow(tmp/tmp1, 0.5);

    printf("iter = %d, error = %g\n", iter, rel_err); 
    if(rel_err < max_err) break;

    iter++;
  }

  printf("end of iterations\n"); 

}



int main(int argc, char* argv[]) {

  char infile[] = "train.csv";
  char outfile[] = "w.csv";
  int digit = 0;  
  //printf("%d, %s\n", atoi(argv[1]), argv[2]);
  checkSamplesCount(infile);
  allocate();
  load_data(infile);
  GradientDescent(0.1, 0.1, 1000, 0.0001);
  output_train_result(outfile);


  return 0;
}
