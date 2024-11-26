#include <iostream>
#include <random>

#include "cellgrids.h"

extern std::random_device rd;

CellGrids::CellGrids(int size) {
    gs = size;
    for(int i=0; i<gs; ++i){
        for(int j=0; j<gs; ++j){
            tc[i][j] = 0; //tumor cell
            m0[i][j] = 0; //M0
            m1[i][j] = 0; //M1
            m2[i][j] = 0; //M2
            allcells[i][j] = 0; //allcells = tc + m1 + m2 + m0 + vas, should be 0 or 1
            vas[i][j] = 0; //vascular cell   
            dead[i][j] = 0;
        }
    }
    
    // set vascular cells (entry points for Macrophages and drug)
    double d;
    std::uniform_real_distribution<> Dis(0.0,1.0);
    
    //random distribution of vascular cells
    //dense outside and sparse inside 
    for(int k=0; k<3; k++){
        for(int i=1; i<gs-1; i++){
            for(int j=1; j<gs-1; j++){
                vas[i][j] = 0;
                allcells[i][j] = 0;
                d = Center_distance(i, j, gs);
                std::mt19937 g(rd());
                if((Dis(g) < 0.09-0.03*k) && (d > 4900 - 1500*k)){
                    vas[i][j] = 1;
                    allcells[i][j] = 1;
                }
            }
        }
    }
    
    // set boundary
    for(int i=0;i<gs;i++){
        for(int j=0;j<gs;j++){
            allcells[i][0] = 1;
            allcells[i][gs-1] = 1;
            allcells[0][j] = 1;
            allcells[gs-1][j] = 1;
        }
    }

}
double CellGrids::Center_distance(int ii, int jj, int size){
    double dd = 0.0;
    dd = (ii-(size-1)/2)*(ii-(size-1)/2)+(jj-(size-1)/2)*(jj-(size-1)/2);
    return dd;
}
