#ifndef CELLGRIDS_H
#define CELLGRIDS_H

class CellGrids {
public:
    CellGrids(int size);
    
    // Static arrays are more efficient than dynamic arrays in terms of runtime performance.
    // Although the input grid size (gs) can be received, it is still necessary to manually 
    // set the static array size equal to the input grid size.
    int tc[100][100]; // tumor cell
    int m0[100][100]; // M0 macrophage
    int m1[100][100]; // M1 macrophage
    int m2[100][100]; // M2 macrophage
    int allcells[100][100]; // number of all cell
    int vas[100][100]; // vascular cell
    int dead[100][100]; // dead cell
    
private:
    int gs; // input grid size
    double Center_distance(int ii, int jj, int size);
};


#endif // CELLGRIDS_H

