#ifndef DEADCELL_H
#define DEADCELL_H

#include <random>

#include "cellgrids.h"
#include "macrophage.h"
#include "diffusibles.h"

class DeadCell{
public:
    DeadCell(std::array<int, 2> loc, std::string Source); // dead cell agent
    std::string source;
    std::array<int,2> location; // (i,j) location
    int removal(CellGrids &cg);
    void simulation(double tstep, CellGrids &cg); // simulate dead cell agent behaviors 
    double removal_time; 
private:
    int contact;
};

#endif // DEADCELL_H
