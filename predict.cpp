#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

int n_feature = 28*28;
int n_sample = 20000;
double **train_mat;
double *result_vec;
double *wvec;
double *yvec, *fvec;



void load_weight(char *filename) {

  int count = 0;
  fstream file;
  file.open(filename);
  string line;
  int row;
  while (getline(file, line,'\n')) {
    istringstream templine(line); 
    string data;
    row = 0;
    while (getline(templine, data,',')) {
      wvec[row] = atof(data.c_str());
      row++;
    }
  }
}



void load_data(char *filename) {

  int count = 0;
  fstream file;
  file.open(filename);
  string line;
  int row = 0;
  int col = 0;
  getline(file, line,'\n');
  while (getline(file, line,'\n')) {
    istringstream templine(line); 
    string data;
    col = 0;
    //cout << row << endl;
    getline(templine, data,',');
    if( atoi(data.c_str()) !=1)
      yvec[row] = 0;
    else
      yvec[row] = 1;   //cout << yvec[row];
   
    //yvec[row] = atoi(data.c_str());
 
    while (getline(templine, data,',')) {
      if(atof(data.c_str()) > 0)
        train_mat[row][col] = 1;
      else
        train_mat[row][col] = 0;
//     cout << train_mat[row][col];
      col++;
//      if(col%28==0) cout << endl;
    }
//    cout <<yvec[row] << "^" <<  endl;
    count++;
    row++;
    if(count >= n_sample) break;
  }
  file.close();

}


int allocate() {
  
  train_mat = (double**)malloc(sizeof(double*)*n_sample);
  result_vec = (double*)malloc(sizeof(double)*n_feature);
  yvec = (double*)malloc(sizeof(double)*n_sample);
  fvec = (double*)malloc(sizeof(double)*n_sample);
  wvec = (double*)malloc(sizeof(double)*n_feature);
  for(int i = 0; i < n_sample; i++) {
    train_mat[i] = (double*)malloc(sizeof(double)*n_feature);
    yvec[i] = 0;
    fvec[i] = 0;
  }
  for(int i = 0; i < n_feature; i++) {
    wvec[i] = 1.0;
  }

}



/* mathematical model
C(w) = 0.5*||f(w)-y||^2
w_{i+1} = w_{i} + a*grad(C(w_{i}))
  where
grad(C)(w) = (f(w) - y)*X
*/


double sigmoid(double z) {

   double tmp = 1/(1 + exp((-1)*z));
//    printf("%g\n", exp((-1)*z)); 
   if(tmp < 0.5) return 0;
   else return 1;
}


void updatefvec() {
  int iter = 0, i, j;
  double tmp;
  for(j = 0; j < n_sample; j++) {

    tmp = 0;
    for(i = 0; i < n_feature; i++) {

      tmp += train_mat[j][i]*wvec[i];
    }

    fvec[j] = sigmoid(tmp); 


  }


}


void calc_error() {

  double tmp = 0; 
  updatefvec();

  for(int i = 0; i < n_sample; i++) {
    tmp  += pow(yvec[i] - fvec[i], 2);


  } 
  printf("n of error = %f\n", tmp);
  printf("%g\n", (n_sample-tmp)/n_sample); 

}



int main(int argc, char* argv[]) {

  char infile[] = "train.csv";
  char outfile[] = "result.csv";
  allocate();
  load_data(infile);
  load_weight(outfile);
  calc_error();
  return 0;
}
