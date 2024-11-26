#include <iostream>
#include <random>

#include "tumor.h"

extern std::random_device rd;

Tumor::Tumor(std::array<int, 2> loc, double prolTime, int index, int initial, double p_tc, double p_d,
             double EGFRI, double IGF1RI, double AKTI, double ERKI, double A_ERK, int TC_max){
  // representation, unit, references of parameters
  // are listed in S2 Table 'Tumor cells' part 
  // and 'Signaling pathways' part 
  state = "alive";
  idx = index;
  location = loc;
  
  age = 0;
  mature = prolTime;
  div_time = 0;
  maxDiv = 8;
  
  // initial = 1 represents intializeCells before 
  // tumor growth simulation, not the initialization
  // of virtual TME
  std::mt19937 g(rd()); 
  std::uniform_real_distribution<>  ageStart(0.0,24*1.0);
  if(initial == 1){ 
    age = ageStart(g);
    mature = prolTime/8;  
    div_time = age;
    maxDiv = 12;
  }
  
  nDiv = 0;
  
  cell_cycle = mature;
  life_span = 24.0 * 8;
  
  if_migra = 0;
  prob = 0.0;
  prob_0 = 0.0;
  
  tc_max = TC_max;
  
  p_D = p_d;
  p_TC = p_tc;
  
  a_TI = 2600; 
  a_TE = 2600;
  D_TC = 1e-9; 
  adjust_v1 = 0.85;
  
  dt = 0.00069; // 1 min = 1/1440 day
  time = 0;
  
  K1 = 0.5;
  K2 = 0.5;
  
  a_AKT = 44;
  a_ERK = A_ERK;
  
  EGFR_max = 1;
  ERK_max = 1;
  AKT_max = 1;
  IGF1R_max = 1;
  
  EGFR = EGFRI;
  IGF1R = IGF1RI;
  AKT = AKTI;
  ERK = ERKI;
  
  V3 = 99.8031372676745;
  K31 = 1;
  K32 = 0.724982430863775;
  K33 = 0.00790513933992095;
  d3 = 1.56078431465490;
  
  V41 = 69.5058823535373;
  V42 = 33.4117647065726;
  V43 = 0.864196307410695;
  K41 = 1;
  K42 = 0.102633043022255;
  K43 = 0.0214285724285716;
  d4 = 1.07819925585235;
  
  V5 = 23.8901960789529;
  K51 = 0.0400211238684488;
  K52 = 0.261304749242371;
  K53 = 0.0200000010000000;
  d5 = 0.753276246188159;
  
  V6 = 16.6823529420039;
  K61 = 0.425704048002236;
  K62 = 0.998406371352485;
  d6 = 0.416521770982161;
  
  n = 10; 
}

// calculate proliferation probability of tumor cell
void Tumor::pro_prob(Diffusibles &diff, std::vector<Tumor> &tc_list){
  
  // Hill functions
  double H1 = ERK/(K1 + ERK);
  double H2 = AKT/(K2 + AKT);
  
  // current number of tumor cells
  double tc_now = tc_list.size();
  
  // calculate probability of proliferation
  if(state == "quiescent"){
    prob = 1.0;
  }
  else{
    prob = p_TC * (1 + a_ERK*H1 + a_AKT*H2)*(1 - tc_now*1.0/tc_max); 
  }
  
  // probability without restriction of maximum capacity number
  // used for testing
  prob_0 = p_TC * (1 + a_ERK*H1 + a_AKT*H2);
  
  // location
  int i = location[0];
  int j = location[1];
  
  // probability should <= 1.0
  if(prob >= 1.0){
    prob = 1.0;
  }
  
  // record probabilities
  diff.prob[i][j] = prob;
  diff.prob_0[i][j] = prob_0;   
}

// simulate prolifetation of tumor cell
void Tumor::proliferation(CellGrids &cg, std::vector<Tumor> &tc_list, Diffusibles &diff, double prolTime){
  
  // (i, j) loaction
  int i = location[0];
  int j = location[1];
  
  // eight latent proliferation directions
  // Moore neighborhood
  int z = 0;
  std::vector<int> ix;
  std::vector<int> jx;
  std::mt19937 g(rd());
  std::uniform_real_distribution<> Dis(0.0,1.0);
  for(int il=-1; il<2; il++){
    for(int jl=-1; jl<2; jl++){
      ix.push_back(il);
      jx.push_back(jl);
      z++;
    }
  }
  
  double probs[z];
  double sum=0;
  for(int q=0; q<z; q++){
    probs[q] = (1 - cg.allcells[i+ix[q]][j+jx[q]]);    
    sum += probs[q];
  }
  
  // simulate proliferation
  double gg = Dis(g);
  if(gg > prob){
    return;
  }
  else if((sum == 0 || tc_list.size() >= tc_max) && gg <= prob){
    state = "quiescent";
    return;
  }
  else if(sum > 0 && tc_list.size() < tc_max && gg <= prob){
    state = "alive";
    
    // normalization
    double norm_probs[z];
    for(int q=0;q<z;q++){norm_probs[q] = probs[q]/(sum);} 
    for(int q=1; q<z; q++){norm_probs[q] = norm_probs[q] + norm_probs[q-1];} 
    
    // choose proliferation direction
    double p = Dis(rd);
    int choice = 0;
    for(double norm_prob : norm_probs){
      if(p > norm_prob){choice++;}
    }
    
    // proliferation, update cell grid and cell list
    int ni = i + ix[choice];
    int nj = j + jx[choice];
    
    cg.allcells[ni][nj] = 1 ;
    cg.tc[ni][nj] = 1;
    if(cell_cycle < prolTime){
      tc_list.push_back(Tumor({ni,nj}, cell_cycle*2, tc_list.size(), 0, p_TC, p_D, 
                              EGFR/2, IGF1R/2, AKT/2, ERK/2, a_ERK, tc_max));
    }
    else{
      tc_list.push_back(Tumor({ni,nj}, prolTime, tc_list.size(), 0, p_TC, p_D, 
                              EGFR/2, IGF1R/2, AKT/2, ERK/2, a_ERK, tc_max));
    }
    
    // division of proteins
    diff.AKT[ni][nj] = AKT/2;
    diff.ERK[ni][nj] = ERK/2;
    diff.IGF1R[ni][nj] = IGF1R/2;
    diff.EGFR[ni][nj] = EGFR/2;
    
    EGFR = EGFR/2;
    IGF1R = IGF1R/2;
    AKT = AKT/2;
    ERK = ERK/2;
    
    nDiv++; // current number of divisions 
    
    // reset
    prob = 0;
    div_time = 0;
  }   
}

// simulate migration of tumor cell
void Tumor::migration(Diffusibles &diff, CellGrids &cg){
  
  // (i, j) location
  int i = location[0];
  int j = location[1];
  
  // five latent choices of migration
  std::array<int, 5> ix={0,1,0,-1,0};
  std::array<int, 5> jx={1,0,-1,0,0};
  
  // EGF and IGF1 concentration from diffusion equation
  double I[5]={0,0,0,0,0};
  double E[5]={0,0,0,0,0};
  for(int q=0; q<5; q++){
    E[q] = diff.EGF[i+ix[q]][j+jx[q]];
    I[q] = diff.IGF1[i+ix[q]][j+jx[q]];
  }
  std::mt19937 g(rd());
  std::uniform_real_distribution<> Dis(0.0,1.0);
  
  double h = diff.dx; // 60um = spot size
  double k = 2250*0.05*(h/0.0015)*(h/0.0015)*adjust_v1; // choose a suitable k
  
  // calculate probability of latent migration choices
  double probs[5]={0,0,0,0,0};
  probs[4] = (1 - 4*k*D_TC/pow(h,2) - k*a_TI*(I[0]+I[1]+I[2]+I[3]-4*I[4])/pow(h,2) - k*a_TE*(E[0]+E[1]+E[2]+E[3]-4*E[4])/pow(h,2)); // remain stationary
  probs[0] = (1 - cg.allcells[i+ix[0]][j+jx[0]]) * (k*D_TC/pow(h,2) - k*a_TI*(I[2]-I[0])/(4*pow(h,2)) - k*a_TE*(E[2]-E[0])/(4*pow(h,2))); // move up
  probs[1] = (1 - cg.allcells[i+ix[1]][j+jx[1]]) * (k*D_TC/pow(h,2) - k*a_TI*(I[3]-I[1])/(4*pow(h,2)) - k*a_TE*(E[3]-E[1])/(4*pow(h,2))); // move right
  probs[2] = (1 - cg.allcells[i+ix[2]][j+jx[2]]) * (k*D_TC/pow(h,2) - k*a_TI*(I[0]-I[2])/(4*pow(h,2)) - k*a_TE*(E[0]-E[2])/(4*pow(h,2))); // move down
  probs[3] = (1 - cg.allcells[i+ix[3]][j+jx[3]]) * (k*D_TC/pow(h,2) - k*a_TI*(I[1]-I[3])/(4*pow(h,2)) - k*a_TE*(E[1]-E[3])/(4*pow(h,2))); // move left
  
  // identify migration direction
  double sum=0;
  for(int q=0; q<5; q++){
    sum += probs[q];
  }
  if(sum == 0){
    return;
  }
  double norm_probs[5];
  for(int q=0;q<5;q++){norm_probs[q] = probs[q]/(sum);} // normalization
  for(int q=1; q<5; q++){norm_probs[q] = norm_probs[q] + norm_probs[q-1];} 
  
  double p = Dis(rd); 
  int choice = 0;
  // choose migration direction
  for(double norm_prob : norm_probs){
    if(p > norm_prob){choice++;}
  }
  int ni = i + ix[choice];
  int nj = j + jx[choice];
  
  // migration and update cell grid
  location[0] = ni;
  location[1] = nj;
  cg.allcells[i][j]=0;
  cg.m0[i][j]=0;
  cg.m1[i][j]=0;
  cg.m2[i][j]=0;
  cg.tc[i][j]=0;
  cg.allcells[ni][nj]=1;
  cg.m1[ni][nj]=0;
  cg.m2[ni][nj]=0;
  cg.m0[ni][nj]=0;
  cg.tc[ni][nj]=1; 
}

// odes representing signal pathways in tumor cell
void Tumor::ODE_solution(Diffusibles &diff, double tstep){
  time = time + tstep;
  
  double t = 30; // tstep = 0.5h = 30 * 1 min
  double maxDif = 0;
  double c = 0;
  
  // loaction
  int i = location[0];
  int j = location[1];
  
  // cytokines EGF and IGF1
  double E = diff.EGF[i][j] / diff.max_EGF / 5;
  double I = diff.IGF1[i][j] / diff.max_IGF1 / 3000;
  
  // drugs from diffusion equations
  double E_R_I = diff.EGFR_I[i][j];
  double I_R_I = diff.IGF1R_I[i][j];
  
  // forward Eular method
  for(int q=0; q<t; q++){
    maxDif = 0;
    double EGF_R_0 = EGFR;
    double ERK_0 = ERK;
    double IGF1_R_0 = IGF1R;
    double AKT_0 = AKT;
    EGFR = EGFR + dt * ( (V3*E)/(K31+E) * 1/(1 + ERK/K32) * 1/(1 + E_R_I/K33) * (EGFR_max - EGFR) - d3 * EGFR);
    
    ERK = ERK + (dt) * ( (1 + V41*pow(EGFR, n))/(pow(K41, n)+pow(EGFR, n)) 
                           * (1 + V42*IGF1R/(K42+IGF1R)) * V43 / (1 + AKT/K43) * (ERK_max -ERK) - d4 * ERK );
    
    IGF1R = IGF1R + dt * ( V5*I/(K51 + I) * 1/(1 + ERK/K52) * 1/(1 + I_R_I/K53) * (IGF1R_max - IGF1R) - d5 * IGF1R );
    
    AKT = AKT + (dt) * ( V6*EGFR/(K61 + EGFR) * (IGF1R)/(K62 + (IGF1R)) * (AKT_max - AKT) - d6 * AKT );
    
    if(EGF_R_0 > 0){
      c = (EGFR - EGF_R_0)/EGF_R_0;
      if(c > maxDif){maxDif = c;}
    }
    if(ERK_0 > 0){
      c = (ERK - ERK_0)/ERK_0;
      if(c > maxDif){maxDif = c;}
    }
    if(IGF1_R_0 > 0){
      c = (IGF1R - IGF1_R_0)/IGF1_R_0;
      if(c > maxDif){maxDif = c;}
    }
    if(AKT > 0){
      c = (AKT - AKT_0)/AKT_0;
      if(c > maxDif){maxDif = c;}
    }
    // break condition
    if(maxDif<0.00001 && q>5){break;}
  }
  
  // record protein concentration
  diff.AKT[i][j] = AKT;
  diff.ERK[i][j] = ERK;
  diff.IGF1R[i][j] = IGF1R;
  diff.EGFR[i][j] = EGFR; 
}

// simulate tumor cells' hebaviors
void Tumor::simulation( double tstep, CellGrids &cg, std::vector<Tumor> &tc_list, Diffusibles &diff, double prolTime){ 
  if(state == "dead"){
    return;
  }
  
  if(state == "alive"){
    age = age + tstep;
    div_time = div_time + tstep;
    if_migra++;
  }
  
  std::mt19937 g(rd());
  std::uniform_real_distribution<> Dis(0.0, 1.0);
  // unnatural death according to a fixed probability or natural death   
  if((Dis(g) < p_D && state == "alive" && age >= mature) || age >= life_span){
    state = "dead";
    return;
  }
  
  // migration
  if(if_migra == 48){
    migration(diff, cg);
    if_migra = 0;
  }
  
  // signal pathway   
  ODE_solution(diff, tstep);
  
  // see if the tumor cell can proliferate
  if(age >= mature && div_time >= cell_cycle && nDiv < maxDiv){
    pro_prob(diff, tc_list);
    proliferation(cg, tc_list, diff, prolTime);
    if(nDiv > 7){
      std::cout<<"nDiv = "<<nDiv<<std::endl;
    }       
  }   
}

