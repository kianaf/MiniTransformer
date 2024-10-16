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

//seems like it is calculating the number of parameters for the weights
# define NWEIGHTS p * (3 * mdim * nheads + ncum) + (nheads > 1 ? nheads : 0) * ncum + 2

// 3 for query, key, value 
// mdim for the number of dimensions for each of query, key, value size (vector or matrix)
// nheads is the number of heads
// 
// 2 is for distance weights W[NWEIGHTS-2] (distance between two positions weight) and W[NWEIGHTS-1] (distance to end weight))


// isn't used anywhere.
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
    double res = b[0]; //FIXME: So we use two times bias?!
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

void cumulate_qkv(double* W, double* b, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec, double *cumvec, int mdim, int nheads, int ncum) 
{
    for (int i = 0; i < nheads; i++)
        qkv(W+p*3*mdim*i, b+3*mdim*i, input, n, p, queryvec+(n-1)*mdim*i, keyvec+(n-1)*mdim*i, valuevec+(n-1)*mdim*i, mdim);

    for (int k = 0; k < ncum; k++) cumvec[k] = 0.0;

    int cumvecpos = 0;


    // This iterates over the rowas of the input
    for (int i = 0; i < n - 1; i++) { 
        if (i > 1) {
            for (int k = 0; k < ncum; k++)
                // here it adds the previous two cumvecs of previous position to here.
                cumvec[cumvecpos + k] = cumvec[cumvecpos - ncum + k];
        }
        
        double distweight = exp(-pow((n-2 - i)*exp(W[NWEIGHTS-1]),5)); // this is for weight no increase contribution if it is towards the end
        
        for (int l = 0; l < nheads; l++) {
            double cursum = 0.0;
            double curwsum = 0.0;
            for (int j = 0; j <= i; j++) {   // mask out the future
                double curq = 0;
                for (int k = 0; k < mdim; k++) {
                    int curoffset = (n-1)*(k + mdim*l);
                    curq += queryvec[i+curoffset]*keyvec[j+curoffset];
                }

                double curw = exp(curq) * exp(-pow((i-j)*exp(W[NWEIGHTS-2]),5)); // this is for attention
                curwsum += curw;
                cursum += curw*valuevec[j+(n-1)*mdim*l];
            }

            if (i > 0) {
                if (nheads == 1) {
                    cumvec[cumvecpos] += distweight*cursum/curwsum;
                } 
                else 
                {
                    for (int k = 0; k < ncum; k++)
                        cumvec[cumvecpos + k] += distweight*W[p*(3*mdim*nheads+ncum) + k*nheads + l]*cursum/curwsum;
                }
            }
        }
        if (i > 0) cumvecpos += ncum;
    }
} 

double pred_j(int j, double *cumvec, double* W, double* b, int n, int p, int mdim, int nheads, int ncum) 
{
    double res = b[3*mdim*nheads+j];

    for (int k = 0; k < ncum; k++)
        res += (W[(p*3)*mdim*nheads+k*p+j]*cumvec[k]);

    return res;
}


void lossFun(double* loss, double* W, double* b, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec, double *cumvec, int mdim, int nheads, int ncum) 
{
    cumulate_qkv(W, b, input, n, p, 
                                 queryvec, keyvec, valuevec, cumvec,
                                 mdim, nheads, ncum);
    
    double delta = 0.0;
    loss[0] = 0.0;

    for (int i = 2; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double pred = pred_j(j, cumvec + (i-2)*ncum, W, b, n, p, mdim, nheads, ncum);
            delta = input[p*i+j] - pred;
            loss[0] += delta * delta;
        }
    }

    for (int i = 0; i < NWEIGHTS; i++) {
        loss[0] += 0.001 * W[i] * W[i];
    }
}

void lossFun_grad(double* W, double* b, double* input, double* d_W, double* d_b, int n, int p, double *queryvec, double *d_queryvec, double *keyvec, double *d_keyvec, double *valuevec, double *d_valuevec, double  *cumvec, double *d_cumvec, int mdim, int nheads, int ncum) {

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
        enzyme_dup,     cumvec, d_cumvec,
        enzyme_const,     mdim,
        enzyme_const,     nheads,
        enzyme_const,     ncum);

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
                if ((k != pos3 && curran > 0.3) || (k == pos3 && seenfirst && seensecond && curran > 0.1)) {
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
    const int nheads = 16;
    const int ncum = 2;

    const int maxlen = 10;
    const int cumveclen = ncum*(maxlen-2);

    /* layout:
    1. parameters for all input dimensions for each of query, key, value within mdims within heads
    2. parameters for all ouput dimensions for each ncum
    3. parameters for the contribution of each head to each ncum
    4. scaling parameter for the attention distance weighting
    5. scaling parameter for the cumulative distance weighting
    */
    const int nweights = NWEIGHTS;

    /* layout:
    1. bias parameters for the output of each of query, key, value within mdims within heads
    1. bias parameters for the output dimension
    */
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
    
    std::vector<double> input;
    std::vector<int> startvec;

    generate_data(input, startvec, n, p, pos1, pos2, pos3, maxlen);

    // This is for baseline1: Bench1 which averages over all timepoints of all the sequences
    double dimave[p] = {0.0};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) dimave[j] += input[p*i+j];
    }
    for (int i = 0; i < p; i++) {
        dimave[i] /= n;
        std::cout << dimave[i] << std::endl;
    }

    // This is for baseline2: Bench2 which averages over all timepoints of all the sequences for pos1 and pos2 and for pos3, 
    // it averages over pos3 if pos2 ==1 and averages over pos3 if pos2 == 0 and then for prediction can also look at pose2


    // pos3ave0: Will hold the average value of pos3 when pos2 == 0.
	// pos3ave1: Will hold the average value of pos3 when pos2 == 1.
	// pos2count0: Keeps count of the number of times pos2 == 0 is encountered.
	// pos2count1: Keeps count of the number of times pos2 == 1 is encountered.


    double pos3ave0 = 0.0;
    double pos3ave1 = 0.0;
    int pos2count0 = 0;
    int pos2count1 = 0;
    
    for (int k = 0; k < startvec.size(); k++) {
        int curn = (k < startvec.size() - 1 ? startvec[k+1] : input.size() / p) - startvec[k];
        double* curinput = input.data() + startvec[k] * p;
        for (int i = 0; i < curn-1; i++) {
            if (curinput[p*i + pos2] == 1.0) {
                pos2count1++;
                pos3ave1 += curinput[p*(i+1) + pos3];
            } else {
                pos2count0++;
                pos3ave0 += curinput[p*(i+1) + pos3];
            }
        }
    }
    pos3ave0 /= pos2count0;
    pos3ave1 /= pos2count1;
    std::cout << "pos3ave0 " << pos3ave0 << " pos3ave1 " << pos3ave1 << std::endl;
    
    double nu = 0.01;

    double d_W[nweights] = {0.0};
    double d_b[nbiases] = {0.0};
    double d_loss = 1.0;

    double queryvec[nqvals] = {0.0};
    double keyvec[nqvals] = {0.0};
    double valuevec[nqvals] = {0.0};
    double cumvec[cumveclen] = {0.0};

    double d_queryvec[nqvals] = {0.0};
    double d_keyvec[nqvals] = {0.0};
    double d_valuevec[nqvals] = {0.0};
    double d_cumvec[cumveclen] = {0.0};



    /*
        Training
    */ 

    for (int i = 0; i < 100; i++) 
    {
        //std::cout << "iteration " << i << std::endl;
        for (int j = 0; j < startvec.size(); j++) {
            int curn = (j < startvec.size() - 1 ? startvec[j+1] : input.size() / p) - startvec[j];
            double* curinput = input.data() + startvec[j] * p;

            //std::cout << curn << std::endl;

            //for (int l = 0; l < curn-2; l++) 
            {
                int l = 0;
                lossFun_grad(W, b, curinput+l*p, d_W, d_b, curn-l, p,
                    queryvec, d_queryvec, keyvec, d_keyvec, valuevec, d_valuevec, cumvec, d_cumvec, mdim, nheads, ncum);

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

        cumulate_qkv(W, b, curinput, curn, p, 
        queryvec, keyvec, valuevec, cumvec, 
        mdim, nheads, ncum);

        for (int j = 0; j < p; j++) {
        //for (int j = pos3; j < pos3+1; j++) {
            double pred = pred_j(j, cumvec+(curn-3)*ncum, W, b, n, p, mdim, nheads, ncum);
            double delta = curinput[p*(curn-1)+j] - pred;
            predloss += delta * delta;

            // ‌Bench1: average of dimensions as benchmark
            
            double dimsum = 0.0;
            for (int k = 0; k < curn - 1; k++) {
                dimsum += curinput[p*k+j];
            }
            double benchdelta = curinput[p*(curn-1)+j] - dimave[j]; //dimsum/(curn-1);
            benchloss += benchdelta * benchdelta;

            double bench2pred = 0.0;
            if (j == pos3) {
                if (curinput[p*(curn-2)+pos2] == 1.0) {
                    bench2pred = pos3ave1;
                } else {
                    bench2pred = pos3ave0;
                }
            } else {
                bench2pred = dimave[j];
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

    /*
        Analysis of the fitted patterns
    */

    predinput[0] = 1; predinput[1] = 0; predinput[2] = 0;
    predinput[3] = 0; predinput[4] = 0; predinput[5] = 0;
    predinput[6] = 0; predinput[7] = 1; predinput[8] = 0;
    

    // I guess it is trying to print the values of the query, key, value and cumvec but changed to s < 0 to have loss values printed
    for (int s = 0; s < 0; s++) {
        std::cout << "#### sequence: " << s << std::endl;
        
        int curn = (s < predstartvec.size() - 1 ? predstartvec[s+1] : predinput.size() / p) - predstartvec[s];
        double* curinput = predinput.data() + predstartvec[s] * p;

        cumulate_qkv(W, b, curinput, curn, p, 
         queryvec, keyvec, valuevec, cumvec, 
         mdim, nheads, ncum);

        for (int i = 0; i < curn; i++) {
            std::cout << "## obs. no.: " << i << std::endl << "obs. val: ";
            for (int k = 0; k < p; k++) std::cout << curinput[p*i+k] << " ";
            std::cout << std::endl << "value: ";
            for (int k = 0; k < nheads; k++) std::cout << valuevec[(curn-1)*k + i] << " ";
            std::cout << std::endl << "query: ";
            for (int k = 0; k < nheads; k++) std::cout << queryvec[(curn-1)*k + i] << " ";
            std::cout << std::endl << "key: ";
            for (int k = 0; k < nheads; k++) std::cout << keyvec[(curn-1)*k + i] << " ";       
            std::cout << std::endl << std::endl;

            if (true) {
                for (int l = 0; l < nheads; l++) {
                    std::cout << "head: " << l << std::endl;
                    std::vector<double> curw;
                    double curwsum = 0.0;
            
                    for (int j = 0; j <= i; j++) {
                        double curq = 0;
                        for (int k = 0; k < mdim; k++) {
                            int curoffset = (curn-1)*(k + mdim*l);
                            curq += queryvec[i+curoffset]*keyvec[j+curoffset];
                        }
                        std::cout << "query res: " << curq << std::endl;
                        double curwval = exp(curq); // * 1/(abs(i-j) + 1);
                        std::cout << "wval: " << curwval << std::endl;
                        curwsum += curwval;
                        curw.push_back(curwval);
                    }
                    for (int j = 0; j <= i; j++) {
                        curw[j] /= curwsum;
                        std::cout << curw[j] << " ";
                    }
                    std::cout << std::endl;
                
                    /* if (nheads == 1) {
                        dimcumvec[0] += cursum/curwsum;
                    } else {
                        for (int j = 0; j < p; j++)
                            dimcumvec[j] += W[p*(3*mdim*nheads) + l*p + j]*cursum/curwsum;
                    }*/
                }
            }
            
        }
    }
    
    return 0;
}
