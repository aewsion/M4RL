#ifndef CELLGRIDS_H
#define CELLGRIDS_H

class CellGrids {
public:
    CellGrids();
    int tc[100][100]; // tumor cell
    int m0[100][100]; // M0 macrophage
    int m1[100][100]; // M1 macrophage
    int m2[100][100]; // M2 macrophage
    int allcells[100][100]; // number of all cell
    int vas[100][100]; // vascular cell
    int dead[100][100]; // dead cell

private:
    double Center_distance(int ii, int jj);
};


#endif // CELLGRIDS_H
