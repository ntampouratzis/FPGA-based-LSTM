//#include "rnn_header.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include "hls_math.h"
#include <ap_fixed.h>
#include <ap_int.h>

#define n 7000
#define m 64
#define s 6
#define m2 128
#define m4 256

#define np 3
#define mp 64
#define sp 6

typedef ap_fixed<14,2> ftype;
typedef ap_fixed<14,6> dtype;
typedef ap_fixed<9,9> intype;

void rnn_LSTM(ftype w_all[m2][m4], ftype b_all[m4], dtype h[m], dtype c[m], dtype inputs[m2]);

void rnn_model(intype signals[n][m], ftype weights_in[m][m], ftype biases_in[m], ftype w_hidd2[m][m], ftype b_hidd2[m],
		ftype w_hidd3[m][m], ftype b_hidd3[m], ftype w_all[m2][m4], ftype b_all[m4],
		ftype w_all2[m2][m4], ftype b_all2[m4], ftype w_out[m][s], ftype b_out[s], int cl[n]){


	dtype X_next[n][m];
	dtype X_prev[n][m];
	dtype X_final[n][s];

	dtype h[m];
	dtype c[m];
	dtype inputs[m2];

	dtype h2[m];
	dtype c2[m];
	dtype inputs2[m2];

	dtype temp1;
	dtype temp2;
	dtype temp3;
	dtype temp4;

#pragma HLS array_partition variable= X_prev cyclic factor=64 dim=2
#pragma HLS array_partition variable= X_next cyclic factor=64 dim=2
#pragma HLS array_partition variable= X_final cyclic factor=6 dim=2
#pragma HLS array_partition variable=X_final cyclic factor=2 dim=1
#pragma HLS array_partition variable= signals cyclic factor=64 dim=2

#pragma HLS array_partition variable= weights_in cyclic factor=64 dim=1
#pragma HLS array_partition variable= weights_in cyclic factor=16 dim=2
#pragma HLS array_partition variable=biases_in cyclic factor=16 dim=1

#pragma HLS array_partition variable= w_hidd2 cyclic factor=64 dim=1
#pragma HLS array_partition variable= w_hidd2 cyclic factor=16 dim=2
#pragma HLS array_partition variable=b_hidd2 cyclic factor=16 dim=1

#pragma HLS array_partition variable= w_hidd3 cyclic factor=64 dim=1
#pragma HLS array_partition variable= w_hidd3 cyclic factor=16 dim=2
#pragma HLS array_partition variable=b_hidd3 cyclic factor=16 dim=1

#pragma HLS array_partition variable= w_all cyclic factor=128 dim=1
#pragma HLS array_partition variable= w_all cyclic factor=16 dim=2
#pragma HLS array_partition variable= b_all cyclic factor=16 dim=1

#pragma HLS array_partition variable= w_all2 cyclic factor=128 dim=1
#pragma HLS array_partition variable= w_all2 cyclic factor=16 dim=2
#pragma HLS array_partition variable= b_all2 cyclic factor=16 dim=1

#pragma HLS array_partition variable= w_out cyclic factor=64 dim=1
#pragma HLS array_partition variable= w_out cyclic factor=6 dim=2
#pragma HLS array_partition variable=b_out cyclic factor=6 dim=1

//#pragma HLS allocation instances=mul limit=2000 operation


	for(int i=0;i<n;i++){
			for(int j=0; j<m; j++){
			#pragma HLS unroll factor=16
				temp1=biases_in[j];
				temp2=0;
				temp3=0;
				temp4=0;
				#pragma HLS pipeline II=1
				for(int k=0;k<m/4;k=k+1){
					temp1=temp1+signals[i][k]*weights_in[k][j];
		    	}
				for(int k=m/4;k<m/2;k=k+1){
					temp2=temp2+signals[i][k]*weights_in[k][j];
				}
				for(int k=m/2;k<3*m/4;k=k+1){
					temp3=temp3+signals[i][k]*weights_in[k][j];
				}
				for(int k=3*m/4;k<m;k=k+1){
					temp4=temp4+signals[i][k]*weights_in[k][j];
				}
				X_prev[i][j]=(dtype)1/(1+(dtype)hls::exp((dtype)(-(temp1+temp2+temp3+temp4))));
			}
		}

		for(int i=0;i<n;i++){
			for(int j=0; j<m; j++){
			#pragma HLS unroll factor=16
				temp1=b_hidd2[j];
				temp2=0;
				temp3=0;
				temp4=0;
				#pragma HLS pipeline II=1
				for(int k=0; k<m/4; k++){
					temp1=temp1+X_prev[i][k]*w_hidd2[k][j];
				}
				for(int k=m/4; k<m/2; k++){
					temp2=temp2+X_prev[i][k]*w_hidd2[k][j];
				}
				for(int k=m/2; k<3*m/4; k++){
					temp3=temp3+X_prev[i][k]*w_hidd2[k][j];
				}
				for(int k=3*m/4; k<m; k++){
					temp4=temp4+X_prev[i][k]*w_hidd2[k][j];
				}
				X_next[i][j]=temp1+temp2+temp3+temp4;
			}
		}

		for(int i=0;i<n;i++){
			for(int j=0;j<m;j++){
				#pragma HLS unroll factor=16
				temp1=b_hidd3[j];
				temp2=0;
				temp3=0;
				temp4=0;
				#pragma HLS pipeline II=1
				for(int k=0;k<m/4;k++){
					temp1=temp1+X_next[i][k]*w_hidd3[k][j];
				}
				for(int k=m/4;k<m/2;k++){
					temp2=temp2+X_next[i][k]*w_hidd3[k][j];
				}
				for(int k=m/2;k<3*m/4;k++){
					temp3=temp3+X_next[i][k]*w_hidd3[k][j];
				}
				for(int k=3*m/4;k<m;k++){
					temp4=temp4+X_next[i][k]*w_hidd3[k][j];
				}
				X_prev[i][j]=temp1+temp2+temp3+temp4;
			}
		}

	//LSTM CELL
#pragma HLS array_partition variable=h cyclic factor=64 dim=1
#pragma HLS array_partition variable=c cyclic factor=64 dim=1
#pragma HLS array_partition variable= inputs cyclic factor=128 dim=1
#pragma HLS array_partition variable=h2 cyclic factor=64 dim=1
#pragma HLS array_partition variable=c2 cyclic factor=64 dim=1
#pragma HLS array_partition variable= inputs2 cyclic factor=128 dim=1

for(int i=0; i<m; i=i+1){
#pragma HLS unroll factor=64
	h[i]=0;
	c[i]=0;
	h2[i]=0;
	c2[i]=0;
}

for(int i=0;i<n;i=i+1)
{
#pragma HLS allocation instances=rnn_LSTM limit=1 function
    for(int j=0;j<m;j=j+1){
	#pragma HLS unroll factor=64
        inputs[j]=X_prev[i][j];
        inputs[m+j]=h[j];
    }
    rnn_LSTM(w_all, b_all, h, c, inputs);
    for(int j=0;j<m;j=j+1){
	#pragma HLS unroll factor=64
        inputs2[j]=h[j];
        inputs2[m+j]=h2[j];
    }
    rnn_LSTM(w_all2, b_all2, h2, c2, inputs2);
    for(int j=0;j<m;j=j+1){
#pragma HLS unroll factor=64
    	X_next[i][j]=h2[j];
    }
}

//LAYER 7
for(int i=0;i<n;i++){
	for(int j=0; j<s; j++){
		#pragma HLS unroll factor=2
		temp1 = b_out[j];
		temp2=0;
		temp3=0;
		temp4=0;
		#pragma HLS pipeline II=1
		for(int k=0; k<m/4; k++){
			temp1=temp1+X_next[i][k]*w_out[k][j];
		}
		for(int k=m/4; k<m/2; k++){
			temp2=temp2+X_next[i][k]*w_out[k][j];
		}
		for(int k=m/2; k<3*m/4; k++){
			temp3=temp3+X_next[i][k]*w_out[k][j];
		}
		for(int k=3*m/4; k<m; k++){
			temp4=temp4+X_next[i][k]*w_out[k][j];
		}
		X_final[i][j] = temp1+temp2+temp3+temp4;
	}

}

	//CALCULATE THE CLASS

    for (int i=0;i<n;i=i+1){
		dtype mx=X_final[i][0];
		int tmp_j;
		bool enter = false;
#pragma HLS pipeline II=1
		for (int j=1;j<s;j++){
			if(X_final[i][j]>mx){
				mx=X_final[i][j];
				enter = true;
				tmp_j = j;
			}
		}
		if(enter)
			cl[i]=tmp_j;
		else
			cl[i]=0;

	}



}

