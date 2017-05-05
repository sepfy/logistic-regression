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
double *new_wvec, *old_wvec;
double *yvec, *fvec;



void output_train_result(char *filename) {

  fstream file;
  file.open(filename, ios::out);
  if(!file){
    cout<<"Fail to open file: "<< filename << endl;
    exit(0);
  }

  for(int i = 0; i < n_feature; i++)
    file << new_wvec[i] <<",";

  file.close();
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
    getline(templine, data,',');
    if( atoi(data.c_str()) !=1)
      yvec[row] = 0;
    else
      yvec[row] = 1;   

 
    while (getline(templine, data,',')) {
      if(atof(data.c_str()) > 0)
        train_mat[row][col] = 1;
      else
        train_mat[row][col] = 0;
      col++;
    }
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
  new_wvec = (double*)malloc(sizeof(double)*n_feature);
  old_wvec = (double*)malloc(sizeof(double)*n_feature);
  for(int i = 0; i < n_sample; i++) {
    train_mat[i] = (double*)malloc(sizeof(double)*n_feature);
    yvec[i] = 0;
    fvec[i] = 0;
  }
  for(int i = 0; i < n_feature; i++) {
    new_wvec[i] = 1.0;
    old_wvec[i] = 1.0;
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
   if(tmp < 0.5) return 0;
   else return 1;
}


void updatefvec() {
  int iter = 0, i, j;
  double tmp;
  for(j = 0; j < n_sample; j++) {

    tmp = 0;
    for(i = 0; i < n_feature; i++) {

      tmp += train_mat[j][i]*old_wvec[i];
    }

    fvec[j] = sigmoid(tmp); 


  }


}





void GradientDescent(double alpha, int max_iter, double max_err) {

  double tmp, tmp1, rel_err;
  int iter = 0, i, j;

  while(iter < max_iter) {
    
    for(i = 0; i < n_feature; i++) {
      old_wvec[i] = new_wvec[i];
    }
 
    updatefvec();
    for(i = 0; i < n_feature; i++) {

      tmp = 0;
      for(j = 0; j < n_sample; j++) {
        tmp += (yvec[j] - fvec[j])*train_mat[j][i];
      }

      new_wvec[i] = old_wvec[i] + alpha*tmp;

    }


    tmp  = 0;
    tmp1 = 0;
    for(i = 0; i < n_feature; i++) {
      tmp  += pow(new_wvec[i] - old_wvec[i], 2);
      tmp1 += pow(new_wvec[i], 2);
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
  char outfile[] = "result.csv";
  allocate();
  load_data(infile);
  GradientDescent(0.01, 200, 0.005);
  output_train_result(outfile);


  return 0;
}
