#include <iostream>
#include <random>

#include "macrophage.h"

extern std::random_device rd;
Macrophage::Macrophage(std::array<int, 2> loc, int initial, std::string State, double k_M12, 
                       double kd_M12, double k_I, int pha, double a_i, double a_c){
    // representation, unit, references of parameters
    // are listed in S2 Table 'Macrophages' part    
    state = State;
    location = loc;
    
    std::uniform_real_distribution<double> life(0.0,1.0);
    std::mt19937 g(rd());
    life_span = 24*30; 
    age = 24 * life(g);
    
    D_M = 1e-9;
    a_MC = 2600*2;
    
    p_M01 = 0.05;
    p_M02 = 0.75;
    
    max_phagocytose = 2;
    now_phagocytose = 0;
    
    K_M12 = k_M12;
    Kd_M12 = kd_M12;
    K_I = k_I;
    a_C = a_c; 
    a_I = a_i;
    
    adjust_v1 = 0.85;
    adjust_v2 = 0.5;
    
    p_M21 = 0.15552/8;
    Pha = pha;
    depletion = 0.01;
    
    trans = 1;
    trans_num = 0;
    rest_time = 0;
    
    // initial = 1 represents intializeCells before 
    // tumor growth simulation, not the initialization
    // of virtual TME
    if(initial == 1) {
        age = life_span*life(g);
        trans_num = rand() % 7;
    } 
}

// simualte migration of TAMs
void Macrophage::migration(CellGrids &cg, Diffusibles &diff){
    // (i,j): location of the TAM
    int i=location[0];
    int j=location[1];
    
    // five latent choices of migration
    std::array<int, 5> ix={0,1,0,-1,0};
    std::array<int, 5> jx={1,0,-1,0,0}; 
    int sum_m = 0;
    for(int q=0; q<4; q++){
        sum_m = sum_m + cg.tc[i+ix[q]][j+jx[q]];
    }
    
    double C[5]={0,0,0,0,0};
    for(int q=0;q<5;q++){
        C[q]=diff.CSF1[i+ix[q]][j+jx[q]]; // CSF1 concentration from diffusion equation
    }
    std::mt19937 g(rd());
    std::uniform_real_distribution<> Dis(0.0,1.0);

    double h = diff.dx; // 60um
    double k = 2250*0.22*(h/0.0015)*(h/0.0015)*adjust_v1; // choose a suitable k
    
    // calculate probability of latent migration choices
    double probs[5]={0,0,0,0,0};
    probs[4] = 1 - 4*k*D_M/pow(h,2) - k*a_MC*(C[0]+C[1]+C[2]+C[3]-4*C[4])/pow(h,2); // remain stationary
    probs[0] = (1 - cg.allcells[i+ix[0]][j+jx[0]]) * (k*D_M/pow(h,2) - k*a_MC*(C[2]-C[0])*adjust_v2/(4*pow(h,2))); // move up
    probs[1] = (1 - cg.allcells[i+ix[1]][j+jx[1]]) * (k*D_M/pow(h,2) - k*a_MC*(C[3]-C[1])*adjust_v2/(4*pow(h,2))); // move right
    probs[2] = (1 - cg.allcells[i+ix[2]][j+jx[2]]) * (k*D_M/pow(h,2) - k*a_MC*(C[0]-C[2])*adjust_v2/(4*pow(h,2))); // move down
    probs[3] = (1 - cg.allcells[i+ix[3]][j+jx[3]]) * (k*D_M/pow(h,2) - k*a_MC*(C[1]-C[3])*adjust_v2/(4*pow(h,2))); // move left
    
    // check positivity of probability
    for(int q=0; q<5; q++){
        if(probs[q] < 0){
            std::cout<<"macro mig prob < 0 !"<<std::endl;
            probs[q] = 0;
        }
    }
    
    // identify and amplify the most likely
    // migration choice (if necessary)
    int maxIdx = 0;
    double maxProb = 0;
    for(int q=0; q<4; q++){
        if(probs[q] > maxProb){
            maxProb = probs[q];
            maxIdx = q;
        }
    }
    probs[maxIdx] = 1.0*probs[maxIdx];
    
    // identify migration direction
    double sum=0;
    for(int q=0; q<5; q++){
        sum += probs[q];
    }
    if(sum == 0){
        return;
    }
    else{
        double norm_probs[5];
        for(int q=0;q<5;q++){norm_probs[q] = probs[q]/(sum);} // normalization
        for(int q=1; q<5; q++){norm_probs[q] = norm_probs[q] + norm_probs[q-1];} 
        
        double p = Dis(g); 
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
        if(state == "M0"){
            cg.m0[i][j] = 0;
            cg.m1[i][j] = 0;
            cg.m2[i][j] = 0;
            cg.m0[ni][nj] = 1;
            cg.allcells[i][j] = 0;
            cg.allcells[ni][nj] = 1;
        }
        else if(state == "M2"){
            cg.m0[i][j] = 0;
            cg.m1[i][j] = 0;
            cg.m2[i][j] = 0;
            cg.m2[ni][nj] = 1;
            cg.allcells[i][j] = 0;
            cg.allcells[ni][nj] = 1;
        }
        else if(state == "M1"){
            cg.m0[i][j] = 0;
            cg.m1[i][j] = 0;
            cg.m2[i][j] = 0;
            cg.m1[ni][nj] = 1;
            cg.allcells[i][j] = 0;
            cg.allcells[ni][nj] = 1;
        }
    }   
}

// simualte differentiaion of M0
void Macrophage::differentiaion(CellGrids &cg){
    // (i,j): location of the TAM
    int i = location[0];
    int j = location[1];
    
    // identify differentiaion by probability
    double prob[3] = {0,0,0};
    prob[0] = p_M01;
    prob[1] = p_M02 + p_M01;
    prob[2] = 1;
    std::mt19937 g(rd());
    std::uniform_real_distribution<> dis(0.0,1.0);
    double p = dis(rd);
    int choice = 0;
    for(int k=0;k<3;k++){
        if(p>=prob[k]){choice++;}
    }
    
    // differentiaion and update cell grid
    if(choice == 0){
        cg.m0[i][j] = 0;
        cg.m1[i][j] = 1;
        state = "M1";
        return;
    }
    else if(choice == 1){
        cg.m0[i][j] = 0;
        cg.m2[i][j] = 1;
        state = "M2";
        return;
    }
    else if(choice == 2){
        return;
    }
    
}

// simualte transformation of M1 & M2
int Macrophage::transformation(CellGrids &cg, Diffusibles &diff){
    
    // (i,j): location of the TAM
    int i = location[0];
    int j = location[1];
    
    // identidy if M1 or M2 can transform at the current location
    int z = 0;
    trans = 1;
    std::vector<int> ix;
    std::vector<int> jx;
    for(int il=-1; il<2; il++){
        for(int jl=-1; jl<2; jl++){
            ix.push_back(il);
            jx.push_back(jl);
            z++;
        }
    }
    for(int q=0; q<z; q++){
        trans = trans*(1 - cg.tc[i+ix[q]][j+jx[q]]);
    }
    
    // Hill function calculating
    double C = diff.CSF1[i][j]/diff.max_CSF1; // normalization
    double C_I = diff.CSF1R_I[i][j]; // drug dose
    
    double H_C = C/(K_M12 + Kd_M12*C_I + C);
    double H_I = (diff.A_0 + diff.I[i][j])/(K_I + diff.A_0 + diff.I[i][j]);
    
    double prob_12 = 0.0; // transformation probability from M1 to M2 
    double prob_21 = 0.0; // transformation probability from M2 to M1
    
    // activation of CSF1R
    if(C_I > 0){CSF1R = 0;}
    
    // transformation and update cell grid
    std::mt19937 g(rd());
    std::uniform_real_distribution<> Dis(0.0, 1.0);
    double p = Dis(g);
    double q = Dis(g);
    if(state == "M0"){return 0;}
    else if(state == "M1"){
        prob_12 = a_C*H_C + a_I*H_I;
        if(p < prob_12){
            state = "M2";
            cg.m1[i][j] = 0;
            cg.m2[i][j] = 1;
        }
        return 1;
    }
    else if(state == "M2"){
        if(diff.if_CSF1R_I == true){
            prob_21 = p_M21; // fixed probability from M2 to M1
        }
        else{
            prob_21 = p_M21;
        }
        if(p < prob_21 ){
            state = "M1";
            cg.m2[i][j] = 0;
            cg.m1[i][j] = 1;
        }
        return 1;
    }
    return 0;
}

// simualte phagocytosis of M1
int Macrophage::phagocytosis(CellGrids &cg, Diffusibles &diff){
    
    // (i,j): location of the TAM
    int i=location[0];
    int j=location[1];
    
    // four latent choices of phagocytosis
    std::array<int, 4> ix={0,1,0,-1};
    std::array<int, 4> jx={1,0,-1,0};
    
    // calculate probability of latent phagocytosis choices
    double C[4]={0,0,0,0};
    for(int q=0;q<4;q++){
        C[q]=diff.CSF1[i+ix[q]][j+jx[q]];
    }
    std::mt19937 g(rd());
    std::uniform_real_distribution<> Dis(0.0,1.0);
    
    double h = diff.dx; 
    double k = 2250*0.22; 
    
    double probs[4]={0,0,0,0};
    probs[0] = (cg.tc[i+ix[0]][j+jx[0]] + cg.dead[i+ix[0]][j+jx[0]]) * (k*D_M/pow(h,2) - k*a_MC*(C[2]-C[0])/(4*pow(h,2)));
    probs[1] = (cg.tc[i+ix[1]][j+jx[1]] + cg.dead[i+ix[1]][j+jx[1]]) * (k*D_M/pow(h,2) - k*a_MC*(C[3]-C[1])/(4*pow(h,2))); 
    probs[2] = (cg.tc[i+ix[2]][j+jx[2]] + cg.dead[i+ix[2]][j+jx[2]]) * (k*D_M/pow(h,2) - k*a_MC*(C[0]-C[2])/(4*pow(h,2)));
    probs[3] = (cg.tc[i+ix[3]][j+jx[3]] + cg.dead[i+ix[3]][j+jx[3]]) * (k*D_M/pow(h,2) - k*a_MC*(C[1]-C[3])/(4*pow(h,2)));
    
    // check positivity of pobability
    for(int q=0; q<4; q++){
        if(probs[q] < 0){
            std::cout<<"pha prob < 0 !"<<std::endl;
            probs[q] = 0;
        }
    }
    
    // identify phagocytosis choice
    double sum=0;
    for(int q=0; q<4; q++){
        sum += probs[q];
    }
    if(sum == 0){return 0;}
    
    double norm_probs[4];
    for(int q=0;q<4;q++){norm_probs[q] = probs[q]/(sum);}
    for(int q=1; q<4; q++){norm_probs[q] = norm_probs[q] + norm_probs[q-1];}
    
    double p = Dis(rd); 
    int choice = 0;
    for(double norm_prob : norm_probs){
        if(p > norm_prob){choice++;}
    }
    
    // phagocytosis and update cell grid
    int ni = i + ix[choice];
    int nj = j + jx[choice];
    
    cg.allcells[i][j] = 0;
    cg.m1[i][j] = 0;  
    cg.m1[ni][nj] = 1;
    if(cg.tc[ni][nj] == 1){
        cg.tc[ni][nj] = 0;
    }
    else if(cg.dead[ni][nj] == 1){
        cg.dead[ni][nj] = 0;
    }
    else{
        std::cout<<"phagocytosis error"<<" ni = "<<ni<<" nj = "<<nj<<std::endl;
    }  
    location[0] = ni;
    location[1] = nj;
    
    return 1; 
}

// simulate TAM agent
void Macrophage::simulaion(double tstep, CellGrids &cg, Diffusibles &diff, double depletion){
    
    // TAM death
    if(state == "dead"){
        return;
    }
    
    // M1 rest time
    if(now_phagocytose >= max_phagocytose){          
        rest_time = rest_time + tstep;
        if(rest_time >= 12.0){
            now_phagocytose = 0;
            rest_time = 0;
        }
        return;
    }
    
    // die from CSF1R_I depletion
    if(diff.drug_time > 0){
        std::uniform_real_distribution<double> dis(0.0,1.0);
        if(dis(rd) < depletion){state = "dead"; return;} 
        
    }
    
    // simualte migration of TAMs
    migration(cg, diff);
    
    // simualte phagocytosis of M1
    for(int i=0; i<Pha; i++){
        if(diff.time > 0*24 && i<Pha-1 && state == "M1" && now_phagocytose < max_phagocytose){
            now_phagocytose = now_phagocytose + phagocytosis(cg, diff);
        } 
    }    
    
    // age 
    age = age + tstep; 
    
    // natural death
    if(age >= life_span){
        state = "dead";
        return;
    }
    
    // simualte differentiaion of M0
    if(state == "M0" && age >= 24*1){
        if(trans_num >= 6){
            differentiaion(cg);
            trans_num = 0;
            return;
        }
        else{
            trans_num++;
            return;
        }
        return;
    }
    
    // simualte transformation of M1 & M2
    if(state == "M2"){
        if(trans_num == 6){
            transformation(cg, diff);
            trans_num = 0;
            return;
        }
        else{
            trans_num++;
            return;
        }
    }
    else if(state == "M1"){
        if(trans_num == 6){
            transformation(cg, diff);
            trans_num = 0;
            return;
        }
        else{
            trans_num++;
            return;
        }
    }
}

