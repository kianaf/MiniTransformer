// main.cpp

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>

template < typename return_type, typename ... T >
return_type __enzyme_autodiff(void*, T ... );

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

void qkv(double* W, double* b, double* input, int n, int p, 
         double *queryvec, double *keyvec, double *valuevec, int mdim) {
    for (int k = 0; k < mdim; k++) {
        for (int i = 0; i < n - 1; i++) {
            queryvec[i+(n-1)*k] = query(W+(p*3)*k,b+3*k,input + i*p,p);
            keyvec[i+(n-1)*k] = key(W+(p*3)*k,b+3*k,input + i*p,p);
            valuevec[i+(n-1)*k] = value(W+(p*3)*k,b+3*k,input + i*p,p);
        }
    }
}

void all_qkv(double* W, double* b, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec, int mdim, int nheads) 
{
    for (int i = 0; i < nheads; i++)
        qkv(W+p*3*mdim*i, b+3*mdim*i, input, n, p, queryvec+(n-1)*mdim*i, keyvec+(n-1)*mdim*i, valuevec+(n-1)*mdim*i, mdim);
}

void cumulate_heads(double* W, double* b, int n, int p, double *queryvec, double *keyvec, double *valuevec, double *dimcumvec, int mdim, int nheads) 
{
    for (int j = 0; j < p; j++) dimcumvec[j] = 0.0;

    for (int i = 0; i < n - 1; i++) {
        for (int l = 0; l < nheads; l++) {
            double cursum = 0.0;
            double curwsum = 0.0;
            for (int j = 0; j <= i; j++) {
                double curq = 0;
                for (int k = 0; k < mdim; k++) {
                    int curoffset = (n-1)*(k + mdim*l);
                    curq += queryvec[i+curoffset]*keyvec[j+curoffset];
                }

                double curw = exp(curq) * 1/(abs(i-j) + 1);
                curwsum += curw;
                cursum += curw*valuevec[j+(n-1)*mdim*l];
            }
            if (nheads == 1) {
                dimcumvec[0] += cursum/curwsum;
            } else {
                for (int j = 0; j < p; j++)
                    dimcumvec[j] += W[p*(3*mdim*nheads) + l*p + j]*cursum/curwsum;
            }
        }
    }
} 

double pred_i(int i, double repcum, double* W, double* b, int n, int p, int mdim, int nheads) 
{
    return (repcum + b[3*mdim*nheads+i]);
}

void lossFun(double* loss, double* W, double* b, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec, double *dimcumvec, int mdim, int nheads) 
{
    all_qkv(W, b, input, n, p, queryvec, keyvec, valuevec, mdim, nheads);
    cumulate_heads(W, b, n, p, queryvec, keyvec, valuevec, dimcumvec, mdim, nheads);
    
    double delta = 0.0;
    loss[0] = 0.0;

    for (int i = 0; i < p; i++) {
        double pred = pred_i(i, dimcumvec[i], W, b, n, p, mdim, nheads);
        delta = input[p*(n-1)+i] - pred;
        loss[0] += delta * delta ;
    }

    for (int i = 0; i < p*3*mdim*nheads+nheads*p; i++) {
        loss[0] += 0.001 * W[i] * W[i];
    }
}

void lossFun_grad(double* W, double* b, double* input, double* d_W, double* d_b, int n, int p, double *queryvec, double *d_queryvec, double *keyvec, double *d_keyvec, double *valuevec, double *d_valuevec, double *dimcumvec, double *d_dimcumvec, int mdim, int nheads) {

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
        enzyme_dup,     valuevec, d_valuevec,
        enzyme_dup,     dimcumvec, d_dimcumvec,
        enzyme_const,     mdim,
        enzyme_const,     nheads);

    //std::cout << "loss " << loss << std::endl;
}

void generate_data(std::vector<double> &input, std::vector<int> &startvec, 
                    int n, int p = 3, int pos1 = 0, int pos2 = 1, int pos3 = 2, int maxlen = 4) {

    int curpos = 0;
    
    for (int i = 0; i < n; i++) {
        //std::cout << "individual " << i << std::endl;
        bool justseenfirst = false;
        bool seenfirst = false;
        bool justseensecond = false;
        bool seensecond = false;
        startvec.push_back(curpos);
        //std::cout << "curpos " << curpos << std::endl;
        for (int j = 0; j < maxlen; j++) {
            curpos++;
            for (int k = 0; k < p; k++) {
                double curran = (double)rand() / RAND_MAX;
                if ((k != pos3 && curran > 0.6) || (k == pos3 && seenfirst && seensecond && curran > 0.2)) {
                    input.push_back(1.0);
                    //std::cout << "1.0 ";
                    if (k == pos3) {
                        justseenfirst = false;
                        justseensecond = false;
                    }
                    if (k == pos1) justseenfirst = true;
                    if (seenfirst && k == pos2) justseensecond = true;
                } else {
                    input.push_back(0.0);
                    //std::cout << "0.0 ";
                }
            }
            //std::cout << std::endl;
            seenfirst = justseenfirst;
            seensecond = justseensecond;
            if (j > 1 && ((double)rand() / RAND_MAX > 0.8)) break;
        }
    }
}

int main() {
    const int n = 200;
    const int p = 3;

    const int mdim = 1;
    const int nheads = 5;

    const int nweights = p*(3*mdim*nheads) + (nheads > 1 ? nheads : 0)*p;
    const int nbiases = (3*mdim)*nheads+p;
    const int nqvals = (n-1)*mdim*nheads;
    
    double W[nweights] = {1.0};
    double b[nbiases] = {-0.0};

    srand(time(NULL));
    
    for (int i = 0; i < nweights; i++) {
        W[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
    }

    for (int i = 0; i < nbiases; i++) {
        b[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
    }

    double aaa = 1.0;
    double bbb = 0.0;
    double ccc = 0.0;
    double ddd = 1.0;
    
    /*std::vector input = {aaa, 0.0, bbb,    // obs 1
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
                         0.0, 0.0, 0.0,    // obs 7
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 1.0, 0.0};

    std::vector startvec = {0, 4, 8, 12, 16, 20, 24};*/

    const int pos1 = 0;
    const int pos2 = 1;
    const int pos3 = 2;
    const int maxlen = 4;
    
    std::vector<double> input;
    std::vector<int> startvec;

    generate_data(input, startvec, n, p, pos1, pos2, pos3, maxlen);
    
    double nu = 0.01;

    double d_W[nweights] = {0.0};
    double d_b[nbiases] = {0.0};
    double d_loss = 1.0;

    double queryvec[nqvals] = {0.0};
    double keyvec[nqvals] = {0.0};
    double valuevec[nqvals] = {0.0};
    double dimcumvec[p] = {0.0};

    double d_queryvec[nqvals] = {0.0};
    double d_keyvec[nqvals] = {0.0};
    double d_valuevec[nqvals] = {0.0};
    double d_dimcumvec[p] = {0.0};

    for (int i = 0; i < 10000; i++)
    {
        //std::cout << "iteration " << i << std::endl;
        for (int j = 0; j < startvec.size(); j++) {
            int curn = (j < startvec.size() - 1 ? startvec[j+1] : input.size() / p) - startvec[j];
            double* curinput = input.data() + startvec[j] * p;

            //std::cout << curn << std::endl;

            for (int l = 0; l < curn-2; l++) {
                lossFun_grad(W, b, curinput+l*p, d_W, d_b, curn-l, p,
                    queryvec, d_queryvec, keyvec, d_keyvec, valuevec, d_valuevec, dimcumvec, d_dimcumvec, mdim, nheads);

                for (int k = 0; k < nweights; k++)  {
                    W[k] -= nu * d_W[k];
                    d_W[k] = 0.0;
                }
    
                for (int k = 0; k < nbiases; k++)  {
                    b[k] -= nu * d_b[k];
                    d_b[k] = 0.0;
                }
            }
        }
    }
        
        std::cout << "W=";
        for (int i = 0; i < nweights; i++) {
            if (nheads > 1 && i % (3*p*mdim) == 0) {
                if (i / (3*p*mdim) < nheads) {
                    std::cout << "\nhead: " << i / (3*p*mdim)+1 << std::endl;
                } else {
                    std::cout << "\nothers:\n";
                }
            } else {
                if (i % p == 0) std::cout << std::endl << "  ";
            }
            std::cout << W[i] << " "; 
        }
        std::cout << "\nb=";
        for (int i = 0; i < nbiases; i++)  
            std::cout << b[i] << " ";
        std::cout << std::endl;
    //}

    //
    //    Prediction
    //

    std::vector<double> predinput;
    std::vector<int> predstartvec;

    const int predn = 1000;

    generate_data(predinput, predstartvec, predn, p, pos1, pos2, pos3, 4);

    double predloss = 0.0;
    double benchloss = 0.0;
    double bench2loss = 0.0;
    
    for (int i = 0; i < predn; i++) {
        int curn = (i < predstartvec.size() - 1 ? predstartvec[i+1] : predinput.size() / p) - predstartvec[i];
        double* curinput = predinput.data() + predstartvec[i] * p;

        all_qkv(W, b, curinput, curn, p, queryvec, keyvec, valuevec, mdim, nheads);
        cumulate_heads(W, b, n, p, queryvec, keyvec, valuevec, dimcumvec, mdim, nheads);

        for (int j = 0; j < p; j++) {
        //for (int j = pos3; j < pos3+1; j++) {
double pred = pred_i(j, dimcumvec[j], W, b, n, p, mdim, nheads);
            double delta = curinput[p*(curn-1)+j] - pred;
            predloss += delta * delta;

            // average of dimensions as benchmark
            
            double dimsum = 0.0;
            for (int k = 0; k < curn - 1; k++) {
                dimsum += curinput[p*k+j];
            }
            double benchdelta = curinput[p*(curn-1)+j] - dimsum/(curn-1);
            benchloss += benchdelta * benchdelta;

            double bench2pred = 0.0;
            if (j == pos3 && curinput[p*(curn-2)+pos2] == 1.0) {
                bench2pred = 1.0;
            } else {
                bench2pred = dimsum/(curn-1);
            }
            double bench2delta = curinput[p*(curn-1)+j] - bench2pred;
            bench2loss += bench2delta * bench2delta;
        }
    }

    predloss /= predn*p;
    benchloss /= predn*p;
    bench2loss /= predn*p;
    
    std::cout << "predloss " << predloss << std::endl;
    std::cout << "benchloss " << benchloss << std::endl;
    std::cout << "bench2loss " << bench2loss << std::endl;

    return 0;
}
