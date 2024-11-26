#include <iostream>
#include <fstream>
#include <random>

#include "environment.h"

extern std::random_device rd;

int main(int argc, char **argv){
    std::string folder = argv[1]; // save folder
    std::string set = argv[2]; // set
    int N = std::stoi(argv[3]); // number of simulations
    double simTime = std::stod(argv[4]); // simulation time
    int rand = std::stoi(argv[5]); // rand = 1 for multiple (population) simulations 
                                   // rand = 0 for single simulation
    double M0RecProb = 0.0000035; // basal recruitment probability of M0 macrophages
    double TCProl = 28.0; // tumor cell mature and division time
    double K_M12 = 0.02; // Michaelis constant for the Hill function of CSF1
    double Kd_M12 = 14.3; // Michaelis constant for the Hill function of CSF1R_I
    double K_I = 5.7; // Michaelis constant for the Hill function of CSF1R_I accumulation
    double p_TC = 0.01; // tumor cell basal proliferation probability
    double p_D = 0.0001; // tumor cell basal death probability
    int pha = 2; // Maximum phagocytosis number of each M1 macrophage before having a rest

    // save output information to a txt file
    std::string str = "mkdir -p "+folder+"/set_" + set;
    const char *command = str.c_str();
    std::system(command);   
    std::streambuf* coutBuf = std::cout.rdbuf();
    std::ofstream of(folder+"/set_"+set+"/out"+set+".txt");
    std::streambuf* fileBuf = of.rdbuf();
    std::cout.rdbuf(fileBuf);

    // set random key parameters
    std::mt19937 g(rd());
    std::normal_distribution<double> Dis_n(0,1); // normal distribution
    double p1 = Dis_n(g);
    double p2 = Dis_n(g);
    double p3 = Dis_n(g); 
    double a_C, a_I, a_ERK;
    a_C = 0.675*(1.0+0.1*p1);
    a_I = 0.45*(1+0.1*p2);
    a_ERK = 14.3*(1+0.1*p3);
    if(rand == 0){
        a_C = 0.675; // adjustment coefficient of Hill function H_C
        a_I = 0.45; // adjustment coefficient of Hill function H_I
        a_ERK = 14.3; // coefficient associated with ERK to promote tumor growth
    }
    if(a_I < 0){
        a_I = 0;
    }
    if(a_C < 0){
        a_C = 0;
    }
    if(a_ERK < 0){
        a_ERK = 0;
    }

    // record running time
    double start = clock();

    for (int i = 0; i < N; i++) {
        std::cout << "************************************" << std::endl;
        std::cout << "simulation: "<< i << " start" <<std::endl;

        // create virtual TME, step size = 0.5 hour
        Environment model(0.5, folder, std::stoi(set), M0RecProb, TCProl,
                     K_M12, Kd_M12, K_I, p_TC, p_D, pha, rand,
                     a_C, a_I, a_ERK);

        // start simulation             
        model.simulate(simTime);

        std::cout << "simulation: "<< i << " end" <<std::endl;
        std::cout << "************************************" << std::endl;
    }

    double stop = clock();

    // running time
    std::cout << "Duration: " << (stop-start)/CLOCKS_PER_SEC << std::endl;    
    of.flush();
    of.close();
    std::cout.rdbuf(coutBuf);
    
    std::cout << "Duration: "+folder+"/set_"+set <<" "<< (stop-start)/CLOCKS_PER_SEC << std::endl;

    return 0;
}
