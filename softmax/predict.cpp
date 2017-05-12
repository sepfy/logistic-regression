#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <math.h>


using namespace std;

int n_cls = 10;
int n_features = 28*28;
int n_samples = 20000;
double **train_mat;
double **wvec;
double **yvec, **fvec;



void load_weight(char *filename) {

  int count = 0;
  fstream file;
  file.open(filename);
  string line;
  int i = 0, j = 0;
  while (getline(file, line,'\n')) {
    istringstream templine(line); 
    string data;
       
    while (getline(templine, data,',')) {
      if(j >= n_features) break;

      wvec[j][i] = atof(data.c_str());
      j++;
    }
    i++;
  }
}



void load_data(char *filename) {

  int count = 0;
  fstream file;
  file.open(filename);
  string line;
  int row = 0;
  int col = 0;
  int start = 20000;

  for(int i = 0; i < start; i++) {
    getline(file, line,'\n');
  }

  while (getline(file, line,'\n')) {
    istringstream templine(line); 
    string data;

    getline(templine, data,',');
    //cout << data << ": ";
    for(int i = 0; i < n_cls; i++) {
      if(atoi(data.c_str()) == i)
        yvec[row][i] = 1;
      else
        yvec[row][i] = 0;
 //     cout << yvec[row][i]<<",";

    }
 //   cout << endl;

    col = 0;
    while (getline(templine, data,',')) {
      if(atof(data.c_str()) > 0)
        train_mat[row][col] = 1;
      else
        train_mat[row][col] = 0;
      col++;
    }

    count++;
    row++;
    if(count >= n_samples) break;
  }

  file.close();

}


int allocate() {

  train_mat = (double**)malloc(sizeof(double*)*n_samples);

  yvec = (double**)malloc(sizeof(double*)*n_samples);
  fvec = (double**)malloc(sizeof(double*)*n_samples);
  wvec = (double**)malloc(sizeof(double*)*n_features);

  for(int i = 0; i < n_samples; i++) {
    train_mat[i] = (double*)malloc(sizeof(double)*n_features);
    yvec[i] = (double*)malloc(sizeof(double)*n_cls);
    fvec[i] = (double*)malloc(sizeof(double)*n_cls);
  }

  for(int i = 0; i < n_features; i++) {
    wvec[i] = (double*)malloc(sizeof(double)*n_cls);
  }


  for(int i = 0; i < n_features; i++) {
    for(int j = 0; j < n_cls; j++) {
      wvec[i][j] = 1.0;
    }
  }

}




void updatefvec() {

  int i, j, k;
  double tmp[10];
  double tmp1 = 0;
  double max = 0;
  int idx;
  for(i = 0; i < n_samples; i++) {
    max = 0;

    tmp1 = 0;
    for(k =0; k < n_cls; k++) {

      tmp[k] = 0;

      for(j = 0; j < n_features; j++) {
        tmp[k] += train_mat[i][j]*wvec[j][k];
      }

      tmp[k] = exp(tmp[k]);
      tmp1 += tmp[k];

      if(tmp[k] > max){
        idx = k;
        max = tmp[k];
      }
    }


    for(k =0; k < n_cls; k++)
      if( k == idx)
        fvec[i][k] = 1;
      else
        fvec[i][k] = 0;
  }

}




void calc_error() {

  double tmp = 0; 
  updatefvec();


  for(int i = 0; i < n_samples; i++) {
    for(int k = 0; k < 10; k++) {
        if(yvec[i][k] != fvec[i][k]) {
          tmp++;
          break;
        }
    }
  } 

  printf("n of error = %f\n", tmp);
  printf("%g\n", (tmp)/n_samples); 

  

}



int main(int argc, char* argv[]) {

  char infile[] = "train.csv";
  char wfile[] = "w.csv";
  allocate();
  load_data(infile);
  load_weight(wfile);
  calc_error();


  return 0;
}
