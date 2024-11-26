#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "cellgrids.h"
#include "macrophage.h"
#include "tumor.h"
#include "deadcell.h"
#include "diffusibles.h"

class Environment{
public:
    Environment(double stepSize, std::string folder, int set, double M0RecProb, double TCProl, double k_M12,
                double kd_M12, double k_I, double p_tc, double p_d, int pha, int rand,
                double a_c, double a_i, double a_erk); // create TME
    void simulate(double days); // run a simulation
    
private:
    void run_Tumor(CellGrids &cg, Diffusibles &diff); // tumor cells simulation
    void run_Macrophage(CellGrids &cg, Diffusibles &diff, double depletion); // TAMs simulation
    void run_DeadCell(CellGrids &cg); // dead cells simulation
    void recruitm0(CellGrids &cg, double rec);  // recruitment of M0 
    void plot(CellGrids &cg, Diffusibles &diff, int s); // plot spatial states of the simulation
    void save(Diffusibles &diff); // save number of cells and average/maximum concentration of cytokines
    void saveall(CellGrids &cg, Diffusibles &diff, int s); // save current virtual TME to csv files
    void clean(CellGrids &cg, Diffusibles &diff);  // clean dead cells from cell grid and cell lists
    void initializeCells(CellGrids &cg, Diffusibles &diff); // initialize cells for tumor growth simulation
    void updateTimeCourses(int s, CellGrids &cg, Diffusibles &diff); // record time courses
    void checkError(int s, CellGrids &cg); // check if the cell gird matches the cell list
    void printStep(Diffusibles &diff); // print simulation information after a time step
    void load(); // load virtual TME from csv files
    void initialization(CellGrids &cg, Diffusibles &diff); // initialize virtual TME

    // variables used in the simulation
    int initial;
    std::string saveDir;
    double TCProlTime;
    double tstep;
    double M0recProb;
    double endTime;
    double K_M12, Kd_M12, K_I;
    double p_TC, p_D;
    int Pha;
    double EGFRi,IGF1Ri,AKTi,ERKi; 
    double a_I;
    double a_C;
    double a_ERK;
    int TC_max;
    int M_max;
    int DC_max;

    // diffusion parameters
    double dx;

    // cell lists
    std::vector<Tumor> tc_list;
    std::vector<Macrophage> mp_list;
    std::vector<DeadCell> dc_list;

    // tumor cell state grid
    int tcs[100][100];

    // cell number sequences
    std::vector<double> times;
    std::vector<int> tc_num;
    std::vector<int> m0_num;
    std::vector<int> m1_num;
    std::vector<int> m2_num;
    std::vector<int> phago_num;
    std::vector<int> dc_num;

    // cytokine and drug concentration sequences
    std::vector<double> IGF1_max;
    std::vector<double> IGF1_avg;
    std::vector<double> EGF_max;
    std::vector<double> EGF_avg;
    std::vector<double> CSF1_max;
    std::vector<double> CSF1_avg;
    std::vector<double> CSF1R_I_max;
    std::vector<double> CSF1R_I_avg;
    std::vector<double> IGF1R_I_max;
    std::vector<double> IGF1R_I_avg;
    std::vector<double> EGFR_I_max;
    std::vector<double> EGFR_I_avg;
    std::vector<double> AKT_max;
    std::vector<double> AKT_avg;
    std::vector<double> ERK_max;
    std::vector<double> ERK_avg;
    std::vector<double> IGF1R_max;
    std::vector<double> IGF1R_avg;
    std::vector<double> EGFR_max;
    std::vector<double> EGFR_avg;

    // arrays used for virtual TME loading
    std::vector<std::vector<std::string>> tc_matrix;
    std::vector<std::vector<std::string>> mp_matrix;
    std::vector<std::vector<std::string>> dc_matrix;
    std::vector<std::vector<double>> CSF1_matrix;
    std::vector<std::vector<double>> IGF1_matrix;
    std::vector<std::vector<double>> EGF_matrix;
    std::vector<std::vector<double>> I_matrix;
    std::vector<std::vector<double>> IGF1R_I_matrix;
    std::vector<std::vector<double>> CSF1R_I_matrix;
    std::vector<std::vector<double>> EGFR_I_matrix;
    std::vector<std::vector<int>> vas_matrix;
};

#endif // ENVIRONMENT_H
