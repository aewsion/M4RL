#include <iostream>

#include "deadcell.h"

extern std::random_device rd;

DeadCell::DeadCell(std::array<int, 2> loc, std::string Source) {
    location = loc;
    source = Source;
    removal_time = 0;
}

int DeadCell::removal(CellGrids &cg){
    // (i,j): location of the dead cell
    int i = location[0];
    int j = location[1];

    // contact degree with TME
    int z = 0;
    std::vector<int> ix;
    std::vector<int> jx;
    for(int il=-1; il<2; il++){
        for(int jl=-1; jl<2; jl++){
            ix.push_back(il);
            jx.push_back(jl);
            z++;
        }
    }
    int sum=0;
    for(int q=0; q<z; q++){ 
        sum = sum + (1 - cg.allcells[i+ix[q]][j+jx[q]]);
    }
    return sum;
}

void DeadCell::simulation(double tstep, CellGrids &cg){
    contact = removal(cg);
    removal_time = removal_time + contact * tstep; 
    if(removal_time >= 24.0){
        source = "removal";
        return;
    }
}
