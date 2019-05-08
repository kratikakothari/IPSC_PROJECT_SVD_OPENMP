#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <omp.h>

#define epsilon 1.e-8
#define num 3

using namespace std;

template <typename T> double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

int main (int argc, char* argv[])
{
  int M,N;
  double temp;
  string T,P,Db;

  double elapsedTime,elapsedTime2;
  timeval start,end,end2;

  // Check number of arguments
  if(argc < 4)
  {
	  cout<<"Please input the size of Matrix and file name containing matrix"<<endl;
	  return 0;
  }

  M = atoi(argv[1]);
  N = atoi(argv[2]);
  ifstream matrixfile(argv[3]);

  // Check that given matrix should be square
  if(M != N)
  {
	  cout<<"Error: Matrix must be square";
	  return 0;
  }

  double **U,**V, **S;
  double alpha, beta, gamma, c, zeta, t,s,sub_zeta, converge;

  int acum = 0;
  int temp1, temp2;
  converge = 1.0;

  // Assign memory to all matrices U, S, V
  U = new double*[N];
  V = new double*[N];
  S = new double*[N];

  for(int i =0; i<N; i++)
  {
  	U[i] = new double[N];
   	V[i] = new double[N];
    S[i] = new double[N];
  }

  if(!(matrixfile.is_open())){
    cout<<"Error: file not found"<<endl;
    return 0;
  }

  // Initialize U matrix with input matrix
  for(int i = 0; i < M; i++)
  {
    for(int j =0; j < N; j++)
    {
      matrixfile >> U[i][j];
    }
  }
  matrixfile.close();

  // Initialize V matrix as identity matrix and S Matrix with zeros
  for(int i=0; i<M;i++)
  {
    for(int j=0; j<N;j++)
    {

      if(i==j)
      {
        V[i][j] = 1.0;
        S[i][j] = 0.0;
      }
      else
      {
        V[i][j] = 0.0;
        S[i][j] = 0.0;
      }
    }
  }


 gettimeofday(&start, NULL);

 /* SVD using Jacobi algorithm (Sequencial)*/
 double conv;
 while(converge > epsilon)
 {
    //convergence
    converge = 0.0;

    //counter of loops
    acum++;

    for(int i = 0; i<M; i++)
    {
      for(int j = i+1; j<N; j++)
      {
          // Initialize alpha, beta , gamma to zero
          alpha = 0.0;
          beta = 0.0;
          gamma = 0.0;

          // Update alpha, beta , gamma as per the formulae
          #pragma omp parallel for num_threads(num) reduction(+:alpha) reduction(+:beta) reduction(+:gamma)
          for(int k = 0; k<N ; k++)
          {
            // int c = omp_get_thread_num();
            // cout<<"thread number1 "<<c<<endl;
            alpha = alpha + (U[k][i] * U[k][i]);
            beta = beta + (U[k][j] * U[k][j]);
            gamma = gamma + (U[k][i] * U[k][j]);
          }

          // Update converge basicaly is the angle between column i and j
          converge = max(converge, abs(gamma)/sqrt(alpha*beta));

          zeta = (beta - alpha) / (2.0 * gamma);
           //compute tan of angle
          t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));
          //extract cos
          c = 1.0 / (sqrt (1.0 + (t*t)));
          //extract sin
          s = c*t;

        //Apply rotations on U and V
        #pragma omp parallel for num_threads(num)
        for(int k=0; k<N; k++)
        {
            // int c = omp_get_thread_num();
            // cout<<"thread number2 "<<c<<endl;
              temp = U[k][i];
              U[k][i] = c*temp - s*U[k][j];
              U[k][j] = s*temp + c*U[k][j];

              temp = V[k][i];
              V[k][i] = c*temp - s*V[k][j];
              V[k][j] = s*temp + c*V[k][j];

        }

      }
    }
 }

 //Create matrix S

 for(int i =0; i<M; i++)
 {

   t=0;
   for(int j=0; j<N;j++)
   {
     t=t + pow(U[i][j],2);
   }
   t = sqrt(t);

   for(int j=0; j<N;j++)
   {
     U[i][j] = U[i][j] / t;
     if(i == j)
     {
       S[i][j] = t;
     }
   }
 }

gettimeofday(&end, NULL);

// Print final matrix U
cout<<"\nMatrix U"<<endl;
for(int i=0; i<M; i++)
{
  for(int j=0; j<N; j++)
    cout<<U[i][j]<<" ";
  cout<<endl;
}

// Print final matrix S
cout<<"\nMatrix S"<<endl;
for(int i=0; i<M; i++)
{
  for(int j=0; j<N; j++)
    cout<<S[i][j]<<" ";
  cout<<endl;
}

// Print final matrix V_t
cout<<"\nMatrix V Transpose"<<endl;
for(int i=0; i<M; i++)
{
  for(int j=0; j<N; j++)
    cout<<V[j][i]<<" ";
  cout<<endl;
}

// Print time and iterations

cout<<"iterations: "<<acum<<endl;
elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;



}
