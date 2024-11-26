#ifndef TUMOR_H
#define TUMOR_H

#include "cellgrids.h"
#include "macrophage.h"
#include "diffusibles.h"

class Tumor{
public:
    Tumor(std::array<int, 2> loc, double prolTime, int index, int initial, double p_tc, double p_d,
          double EGFRI, double IGF1RI, double AKTI, double ERKI, double A_ERK, int TC_max); // Tumor cell agent
    void proliferation(CellGrids &cg, std::vector<Tumor> &tc_list, Diffusibles &diff, double prolTime); // simulate prolifetation of tumor cell
    void pro_prob(Diffusibles &diff, std::vector<Tumor> &tc_list); // calculate proliferation probability of tumor cell
    void migration(Diffusibles &diff, CellGrids &cg); // simulate migration of tumor cell
    void ODE_solution(Diffusibles &diff, double tstep); // odes representing signal pathways in tumor cell
    void simulation(double tstep, CellGrids &cg, std::vector<Tumor> &tc_list, Diffusibles &diff, double prolTime); // simulate tumor cells' hebaviors
    
    std::string state; // tumor cell state: active or quiescent
    int idx; // index
    std::array<int,2> location; // (i,j) location
    
    double age; // age
    double mature; // mature time
    double div_time; // time after the latest division
    int maxDiv; // maximum number of division
    int nDiv; // current number of division
    
    double cell_cycle; // cell cycle
    double life_span; // lifespan
    
    int if_migra; // used to control tumor cell migration interval
    double prob; // tumor cell proliferate probability restricted by maximum capacity
    double prob_0; // proliferation probability without maximum capacity restriction
    
    // protein concentrations in tumor cell signaling pathways 
    double EGFR; 
    double IGF1R; 
    double AKT;
    double ERK; 
    
    double a_ERK; // coefficient associated with ERK to promote tumor growth rate
    double a_AKT; // coefficient associated with AKT to promote tumor growth rate 
    
private:
    double tc_max; // tumor cell maximum capacity number in the TME
    
    double p_TC; // tumor cell basal proliferation probability
    double p_D; // tumor cell basal death probability
    
    double a_TI; // chemotaxis coefficient of IGF1 for tumor cells
    double a_TE; // chemotaxis coefficient of EGF for tumor cells
    double D_TC; // diffusion coefficient of tumor cells
    
    // maximum protein concentrations in tumor cell signaling pathways 
    double EGFR_max;
    double ERK_max;
    double AKT_max;
    double IGF1R_max; 
    
    double dt; // dt for solving odes
    double time; // record simulation time
    
    // Michaelis constant for Hill functions in tumor cell
    // proliferation probability calculating
    double K1, K2;
    
    // parameters in tumor cell signal pathways
    double V3, K31, K32, K33, d3;
    double V41, V42, V43, K41, K42, K43, d4; 
    double V5, K51, K52, K53, d5;
    double V6, K61, K62, d6;
    double n; 
    
    double adjust_v1; // adjustment coefficient due to cell spot size
};

#endif // TUMOR_H
