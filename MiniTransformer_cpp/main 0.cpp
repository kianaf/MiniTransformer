// main.cpp

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>

// extern double __enzyme_autodiff(void*, double);
// extern "C" {
    template < typename return_type, typename ... T >
    return_type __enzyme_autodiff(void*, T ... );
// }

// double __enzyme_autodiff(void*,...);

int enzyme_dupnoneed;
int enzyme_dup;
int enzyme_const;


double model(double* W, double* b, double* obs, int p) {
    double res = b[0];
    for(int i=0; i<p; i++)
        res += W[i] * obs[i];

    return res;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double activation(double* W, double* b, double* obs, int p) {
    double res = b[0];
    for (int i=0; i<p; i++)
        res += relu(obs[i] * W[i] + b[0]);

    return res;
}

double query(double* W, double* b, double* obs, int p) {
    return activation(W, b, obs, p);
}

double key(double* W, double* b, double* obs, int p) {
    return activation(W+p, b+1, obs, p);
}

double value(double* W, double* b, double* obs, int p) {
    return activation(W+(p*2), b+2, obs, p);
}

void lossFun(double* loss, double* W, double* b, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec) {
    
    for (int i = 0; i < n - 1; i++) {
        queryvec[i] = query(W,b,input + i*p,p);
        keyvec[i] = key(W,b,input + i*p,p);
        valuevec[i] = value(W,b,input + i*p,p);
    }

    double repcum = 0.0;
    
    for (int i = 0; i < n - 1; i++) {
        double cursum = 0.0;
        double curwsum = 0.0;
        for (int j = 0; j < n - 1; j++) {
            double curw = exp(queryvec[i]*keyvec[j]) * 1/(abs(i-j) + 1);

            curwsum += curw;
            cursum += curw*valuevec[j];
        }
        repcum += cursum/curwsum;
    }
    
    double delta = 0.0;
    loss[0] = 0.0;

    for (int i = 0; i < p; i++) {
        double pred = (W[p*3+i]*repcum + b[3+i]);        
        delta = input[p*(n-1)+i] - pred;
        loss[0] += delta * delta ;
    }
}

void lossFun_grad(double* W, double* b, double* input, double* d_W, double* d_b, int n, int p, double *queryvec, double *d_queryvec, double *keyvec, double *d_keyvec, double *valuevec, double *d_valuevec) {

    double loss;
    double d_loss = 1.0;

    __enzyme_autodiff<void>((void*)lossFun,
        enzyme_dup, &loss, &d_loss,
        enzyme_dup,       W, d_W,
        enzyme_dup,       b, d_b,
        enzyme_const,     input,
        enzyme_const,     n,
        enzyme_const,     p,
        enzyme_dup,     queryvec, d_queryvec,
        enzyme_dup,     keyvec, d_keyvec,
        enzyme_dup,     valuevec, d_valuevec);

    //std::cout << "loss " << loss << std::endl;
}

int main() {
    const int n = 4;
    const int p = 3;
    
    double W[p*4] = {1.0};
    double b[3+p] = {-7.0};

    srand(time(NULL));
    
    for (int i = 0; i < p*4; i++) {
        W[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
    }

    for (int i = 0; i < 4; i++) {
        b[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
    }

    double aaa = 1.0;
    double bbb = 0.0;
    double ccc = 0.0;
    double ddd = 1.0;
    
    std::vector <double> input = {aaa, 0.0, bbb,    // obs 1, startvec[j] = 0
                         0.0, 1.0, 0.0, 
                         0.0, 0.0, 0.0, 
                         ccc, 0.0, ddd,
                         aaa, 0.0, bbb,    // obs 2
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         ccc, 0.0, ddd,
                         0.0, 0.0, 0.0,    // obs 3
                         aaa, 0.0, bbb,
                         0.0, 1.0, 0.0,
                         ccc, 0.0, ddd,
                         aaa, 0.0, bbb,    // obs 4
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         aaa, 0.0, bbb,    // obs 5
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0,
                         ccc, 0.0, ddd,
                         0.0, 0.0, 0.0,    // obs 6
                         aaa, 0.0, bbb,
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0,    // obs 7,  startvec[j] = 24
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 1.0, 0.0};

    std::vector<int> startvec = {0, 4, 8, 12, 16, 20, 24};
    
    double nu = 0.01;

    double d_W[p*4] = {0.0};
    double d_b[3+p] = {0.0};
    double d_loss = 1.0;

    double queryvec[n-1] = {0.0};
    double keyvec[n-1] = {0.0};
    double valuevec[n-1] = {0.0};

    double d_queryvec[n-1] = {0.0};
    double d_keyvec[n-1] = {0.0};
    double d_valuevec[n-1] = {0.0};

    for (int i = 0; i < 1000; i++)
    {
        //std::cout << "iteration " << i << std::endl;

        // startvec.size() = 7 (the number of observations)
        // input.size() = 84 (the number of elements in input)
        
        for (int j = 0; j < startvec.size(); j++) {
            // if j is the last element of startvec size: curn = input.size() / p - startvec[j]
            // else curn = startvec[j+1] - startvec[j]
            // curn = number of obs?


            //input.size() / p = 28
            int curn = (j < startvec.size() - 1 ? startvec[j+1] : input.size() / p) - startvec[j];
            
            //pointer to sequence start 
            double* curinput = input.data() + startvec[j] * p; 

            //std::cout << curn << std::endl;

            for (int l = 0; l < curn-2; l++) {
                lossFun_grad(W, b, curinput+l*p, d_W, d_b, curn-l, p,
                    queryvec, d_queryvec, keyvec, d_keyvec, valuevec, d_valuevec);

                for (int k = 0; k < p*4; k++)  {
                    W[k] -= nu * d_W[k];
                    d_W[k] = 0.0;
                }
    
                for (int k = 0; k < 3+p; k++)  {
                    b[k] -= nu * d_b[k];
                    d_b[k] = 0.0;
                }
            }
        }
    }
        
        std::cout << "W=";
        for (int i = 0; i < p*4; i++)  
            std::cout << W[i] << " "; 
        std::cout << ", b=";
        for (int i = 0; i < 3+p; i++)  
            std::cout << b[i] << " ";
        std::cout << std::endl;
    //}
}
