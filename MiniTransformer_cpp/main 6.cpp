// main.cpp

#include <iostream>
#include <cmath>
#include <vector>
//#include <omp.h>

template < typename return_type, typename ... T >
return_type __enzyme_autodiff(void*, T ... );

int enzyme_dupnoneed;
int enzyme_dup;
int enzyme_const;


#define NWEIGHTS p*(3*mdim*nheads+ncum)+(nheads > 1 ? nheads : 0)*ncum+2
#define NBIASES (3*mdim)*nheads+p

struct ModelArch {
    const int p;
    const int mdim;
    const int nheads;
    const int ncum;
    const int maxlen;
    int nweights() { return NWEIGHTS; }
    int nbiases() { return NBIASES; }
};

struct HeadDim {
    double* W_query;
    double* W_key;
    double* W_value;
    double* b_query;
    double* b_key;
    double* b_value;
};

struct Head {
    HeadDim* dim;
};

struct Cumulant {
    double* W_head;
    double* W_out;
};

enum PInit {
    PINIT_ZERO,
    PINIT_PARAM,
    PINIT_GRADIENT
};

struct ModelParams {
    ModelArch& arch;
    /* layout:
    1. parameters for all input dimensions for each of query, key, value within mdims within heads
    2. parameters for all ouput dimensions for each ncum
    3. parameters for the contribution of each head to each ncum
    4. scaling parameter for the attention distance weighting
    5. scaling parameter for the cumulative distance weighting
    */    double *W;
    /* layout:
    1. bias parameters for the output of each of query, key, value within mdims within heads
    1. bias parameters for the output dimension
    */
    double *b;
    double* b_out;

    Head* head;
    Cumulant* cum;

    void zero() {
        for (int i = 0; i < arch.nweights(); i++) W[i] = 0.0;
        for (int i = 0; i < arch.nbiases(); i++) b[i] = 0.0;
    }

    ModelParams(ModelArch& arch, PInit pinit = PINIT_PARAM): arch(arch) {
        W = new double[arch.nweights()];
        b = new double[arch.nbiases()];

        b_out = b + 3*arch.mdim*arch.nheads;
        
        if (pinit == PINIT_PARAM) {
            srand(time(NULL));
            for (int i = 0; i < arch.nweights(); i++) W[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
            for (int i = 0; i < arch.nbiases(); i++) b[i] = ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
        }
    
        if (pinit == PINIT_GRADIENT || pinit == PINIT_ZERO) {
            zero();
        }
        
        head = new Head[arch.nheads];
        double *curpos_W = W;
        double *curpos_b = b;
        for (int i = 0; i < arch.nheads; i++) {
            head[i].dim = new HeadDim[arch.mdim];
            for (int j = 0; j < arch.mdim; j++) {
                head[i].dim[j].W_query = curpos_W;
                curpos_W += arch.p;
                head[i].dim[j].W_key = curpos_W;
                curpos_W += arch.p;
                head[i].dim[j].W_value = curpos_W;
                curpos_W += arch.p;

                head[i].dim[j].b_query = curpos_b++;
                head[i].dim[j].b_key = curpos_b++;
                head[i].dim[j].b_value = curpos_b++;
            }
        }

        cum = new Cumulant[arch.ncum];
        double* curpos_W_out = curpos_W;
        curpos_W += arch.p*arch.ncum;
        for (int i = 0; i < arch.ncum; i++) {
            cum[i].W_head = curpos_W;
            curpos_W += arch.nheads;
            cum[i].W_out = curpos_W_out;
            curpos_W_out += arch.p;
        }
    }

    ~ModelParams() {
        delete[] W;
        delete[] b;
    }

    double pairdistpen() const { return W[arch.nweights() - 2]; }
    double preddistpen() const { return W[arch.nweights() - 1]; }

    void update(ModelParams& d_params, double nu) {
        for (int k = 0; k < arch.nweights(); k++)  {
            W[k] -= nu * d_params.W[k];
        }

        for (int k = 0; k < arch.nbiases(); k++)  {
            b[k] -= nu * d_params.b[k];
       }
    }
};

std::ostream &operator<<(std::ostream &os, ModelParams const &m) {
    ModelArch& arch = m.arch;
    for (int i = 0; i < arch.nheads; i++) {
        os << "\nHead " << i+1 << ":" << std::endl;
        os << "Query: " << std::endl;
        for (int j = 0; j < arch.mdim; j++) {
            os << "mdim " << j+1 << ": W: ";
            for (int k = 0; k < arch.p; k++) os << m.head[i].dim[j].W_query[k] << " "; 
            os << " b: " << m.head[i].dim[j].b_query[0] << std::endl;       
        }
        os << "Key: " << std::endl;
        for (int j = 0; j < arch.mdim; j++) {
            os << "mdim " << j+1 << ": W: ";
            for (int k = 0; k < arch.p; k++) os << m.head[i].dim[j].W_key[k] << " "; 
            os << " b: " << m.head[i].dim[j].b_value[0] << std::endl;       
        }
        os << "Value: " << std::endl;
        for (int j = 0; j < arch.mdim; j++) {
            os << "mdim " << j+1 << ": W: ";
            for (int k = 0; k < arch.p; k++) os << m.head[i].dim[j].W_value[k] << " "; 
            os << " b: " << m.head[i].dim[j].b_value[0] << std::endl;       
        }   
        os << "Cum W: ";
        for (int j = 0; j < arch.ncum; j++) {
            os << m.cum[j].W_head[i] << " ";
        }
        os << std::endl;
    }

    os << std::endl;
    
    for (int i = 0; i < arch.p; i++) {
        os << "Dim: " << i+1 << ": ";
        for (int j = 0; j < arch.ncum; j++) os << m.cum[j].W_out[i] << " ";
        os << " b: " << m.b_out[i] << std::endl;
    }

    os << "Pair dist penalty: " << m.pairdistpen() << std::endl;
    os << "Pred dist penalty: " << m.preddistpen() << std::endl;
    
    return os;
}

struct ModelScratch {
    double *queryvec, *keyvec, *valuevec;
    double *cumvec; 

    ModelScratch(ModelArch& arch) {
        const int cumveclen = arch.ncum*(arch.maxlen-2);
        const int nqvals = (arch.maxlen-1)*arch.mdim*arch.nheads;
        
        queryvec = new double[nqvals]{0.0};
        keyvec = new double[nqvals]{0.0};
        valuevec = new double[nqvals]{0.0};

        cumvec = new double[cumveclen]{0.0};
    }

    ~ModelScratch() {
        delete[] queryvec;
        delete[] keyvec;
        delete[] valuevec;
        delete[] cumvec;
    }
};

double relu(double x) {
    return x > 0 ? x : 0;
}

double activation(double* W, double* b, double* obs, int p) {
    double res = b[0];
    for (int i=0; i<p; i++)
        res += relu(obs[i] * W[i] + b[0]);

    return res;
}

void qkv(Head& head, double* input, int n, int p, 
         double *queryvec, double *keyvec, double *valuevec, int mdim) {
    for (int k = 0; k < mdim; k++) {
        for (int i = 0; i < n - 1; i++) {
            queryvec[i+(n-1)*k] = activation(head.dim[k].W_query,head.dim[k].b_query,input + i*p,p);
            keyvec[i+(n-1)*k] = activation(head.dim[k].W_key,head.dim[k].b_key,input + i*p,p);
            valuevec[i+(n-1)*k] = activation(head.dim[k].W_value,head.dim[k].b_value,input + i*p,p);
        }
    }
}

void cumulate_qkv(ModelParams& params, double* input, int n, int p, double *queryvec, double *keyvec, double *valuevec, double *cumvec) 
{
    ModelArch& arch = params.arch;
    
    double* W = params.W;
    double* b = params.b;
    
//#pragma omp parallel for
    for (int i = 0; i < arch.nheads; i++)
        qkv(params.head[i], input, n, p, queryvec+(n-1)*arch.mdim*i, keyvec+(n-1)*arch.mdim*i, valuevec+(n-1)*arch.mdim*i, arch.mdim);

    for (int k = 0; k < arch.ncum; k++) cumvec[k] = 0.0;

    int cumvecpos = 0;
    
    for (int i = 0; i < n - 1; i++) {
        if (i > 1) {
            for (int k = 0; k < arch.ncum; k++)
                cumvec[cumvecpos + k] = cumvec[cumvecpos - arch.ncum + k];
        }
        
        double distweight = exp(-pow((n-2 - i)*exp(params.preddistpen()),5));
        
        for (int l = 0; l < arch.nheads; l++) {
            double cursum = 0.0;
            double curwsum = 0.0;
            for (int j = 0; j <= i; j++) {
                double curq = 0;
                for (int k = 0; k < arch.mdim; k++) {
                    int curoffset = (n-1)*(k + arch.mdim*l);
                    curq += queryvec[i+curoffset]*keyvec[j+curoffset];
                }

                double curw = exp(curq) * exp(-pow((i-j)*exp(params.pairdistpen()),5));
                curwsum += curw;
                cursum += curw*valuevec[j+(n-1)*arch.mdim*l];
            }

            if (i > 0) {
                if (arch.nheads == 1) {
                    cumvec[cumvecpos] += distweight*cursum/curwsum;
                } else {
                    for (int k = 0; k < arch.ncum; k++)
                        cumvec[cumvecpos + k] += distweight*params.cum[k].W_head[l]*cursum/curwsum;
                }
            }
        }
        if (i > 0) cumvecpos += arch.ncum;
    }
} 

void cumulate(ModelParams& params, double* input, int n, ModelScratch& scratch) 
{
    cumulate_qkv(params, input, n, params.arch.p, scratch.queryvec, scratch.keyvec, scratch.valuevec, scratch.cumvec);
}

double pred_dim(int dimno, ModelParams& params, double *cumvec)
{  
    double res = params.b_out[dimno];

    for (int k = 0; k < params.arch.ncum; k++)
        res += (params.cum[k].W_out[dimno]*cumvec[k]);

    return res;
}

double predictAt(int n, int dimno, ModelParams& params, ModelScratch& scratch) {
    return pred_dim(dimno, params, scratch.cumvec+(n-3)*params.arch.ncum);
}


void lossFun(double* loss, ModelParams& params, double* input, int n, int p, ModelScratch& scratch)
{
    ModelArch& arch = params.arch;
    
    cumulate_qkv(params, input, n, p, scratch.queryvec, scratch.keyvec, scratch.valuevec, scratch.cumvec);
    
    double delta = 0.0;
    loss[0] = 0.0;

    for (int i = 2; i < n; i++) {
        for (int j = 0; j < p; j++) {
            delta = input[p*i+j] - predictAt(i+1, j, params, scratch);
            loss[0] += delta * delta;
        }
    }

    for (int i = 0; i < arch.nweights(); i++) {
        loss[0] += 0.001 * params.W[i] * params.W[i];
    }
}

void lossFun_grad(ModelParams* params, ModelParams* d_params, double* input, int n, int p, 
                    ModelScratch* scratch, ModelScratch* d_scratch) {
    
    double loss;
    double d_loss = 1.0;

    __enzyme_autodiff<void>((void*)lossFun,
        enzyme_dup, &loss, &d_loss,
        enzyme_dup,       params, d_params,
        enzyme_const,     input,
        enzyme_const,     n,
        enzyme_const,     p,
        enzyme_dup,       scratch, d_scratch);

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
    //omp_set_num_threads(2);

    //
    //    Generate data  
    //
    
    const int n = 100;
    const int p = 3;
    const int maxlen = 5;

    const int pos1 = 0;
    const int pos2 = 1;
    const int pos3 = 2;
    
    std::vector<double> input;
    std::vector<int> startvec;

    generate_data(input, startvec, n, p, pos1, pos2, pos3, maxlen);

    //
    //    Set up the model
    //
    
    const int mdim = 1;
    const int nheads = 16;
    const int ncum = 2;
    
    ModelArch arch{p,mdim,nheads,ncum,maxlen}; 
    ModelParams params(arch, PINIT_PARAM);
    ModelParams d_params(arch, PINIT_GRADIENT);

    ModelScratch scratch(arch);
    ModelScratch d_scratch(arch);

    
    //
    //    Estimate
    //

    int nepochs = 100;
    double nu = 0.01;
    
    for (int i = 0; i < nepochs; i++)
    {
        //std::cout << "iteration " << i << std::endl;
        for (int j = 0; j < startvec.size(); j++) {
            int curn = (j < startvec.size() - 1 ? startvec[j+1] : input.size() / p) - startvec[j];
            double* curinput = input.data() + startvec[j] * p;

            lossFun_grad(&params, &d_params, curinput, curn, p, &scratch, &d_scratch);

            params.update(d_params, nu);
            d_params.zero();
        }
    }
        
    std::cout << params << std::endl;
    

    //
    //    Predict
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

        cumulate(params, curinput, curn, scratch);

        for (int j = 0; j < p; j++) {
            double delta = curinput[p*(curn-1)+j] - predictAt(curn, j, params, scratch);
            predloss += delta * delta;
        }
    }
            
    predloss /= predn*p;
    std::cout << "predloss " << predloss << std::endl;


    //
    //     Benchmarks
    //
    
    double dimave[p] = {0.0};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) dimave[j] += input[p*i+j];
    }
    for (int i = 0; i < p; i++) {
        dimave[i] /= n;
        //std::cout << dimave[i] << std::endl;
    }

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
    //std::cout << "pos3ave0 " << pos3ave0 << " pos3ave1 " << pos3ave1 << std::endl;
    for (int i = 0; i < predn; i++) {
        int curn = (i < predstartvec.size() - 1 ? predstartvec[i+1] : predinput.size() / p) - predstartvec[i];
        double* curinput = predinput.data() + predstartvec[i] * p;

        for (int j = 0; j < p; j++) {
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

    benchloss /= predn*p;
    bench2loss /= predn*p;
    
    std::cout << "benchloss " << benchloss << std::endl;
    std::cout << "bench2loss " << bench2loss << std::endl;

            
    //
    //    Analysis of the fitted patterns
    //

    predinput[0] = 1; predinput[1] = 0; predinput[2] = 0;
    predinput[3] = 0; predinput[4] = 0; predinput[5] = 0;
    predinput[6] = 0; predinput[7] = 1; predinput[8] = 0;
    
    for (int s = 0; s < 0; s++) {
        std::cout << "#### sequence: " << s << std::endl;
        
        int curn = (s < predstartvec.size() - 1 ? predstartvec[s+1] : predinput.size() / p) - predstartvec[s];
        double* curinput = predinput.data() + predstartvec[s] * p;

        cumulate(params, curinput, curn, scratch);

        for (int i = 0; i < curn; i++) {
            std::cout << "## obs. no.: " << i << std::endl << "obs. val: ";
            for (int k = 0; k < p; k++) std::cout << curinput[p*i+k] << " ";
            std::cout << std::endl << "value: ";
            for (int k = 0; k < nheads; k++) std::cout << scratch.valuevec[(curn-1)*k + i] << " ";
            std::cout << std::endl << "query: ";
            for (int k = 0; k < nheads; k++) std::cout << scratch.queryvec[(curn-1)*k + i] << " ";
            std::cout << std::endl << "key: ";
            for (int k = 0; k < nheads; k++) std::cout << scratch.keyvec[(curn-1)*k + i] << " ";       
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
                            curq += scratch.queryvec[i+curoffset]*scratch.keyvec[j+curoffset];
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
