#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <time.h>
#include <cmath>

using namespace std;

int k;
int kPrime;

// Print matrix that is passed in
void printMatrix(vector<vector<int>> mat) {

    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[i].size(); j++) {
            cout << mat[i][j] << " ";
        }

        cout << endl;
    }


    cout << endl;

}

// Generate matrix with size of n (dim) with random values
vector<vector<int>> genMatrix(int dim) {

    // Generate random values for main matrices

    vector<vector<int>> randVect (dim, vector<int>(dim, 0));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            // cout << "DEBUG 2" << endl;
            randVect[i][j] = rand() % 10;
        }
    }

    return randVect;

}

vector<vector<int>> getSubMat(int offsetX, int offsetY, int dim, vector<vector<int>> mat) {

    int m = dim / 2;

    vector<vector<int>> subMatrix (m, vector<int>(m, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            subMatrix[i][j] = mat[i + offsetX][j + offsetY];
        }
    }

    return subMatrix;

    
}

vector<vector<int>> addMat (vector<vector<int>> x, vector<vector<int>> y, int dim) {

    vector<vector<int>> sumMat (dim, vector<int>(dim, 0));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            sumMat[i][j] = x[i][j] + y[i][j];
        }
    }
    

    return sumMat;
}

vector<vector<int>> subMat (vector<vector<int>> x, vector<vector<int>> y, int dim) {

    vector<vector<int>> diffMat (dim, vector<int>(dim, 0));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            diffMat[i][j] = x[i][j] - y[i][j];
        }
    }
    

    return diffMat;
 
}

vector<vector<int>> combMat(vector<vector<int>> c11, vector<vector<int>> c12, vector<vector<int>> c21, vector<vector<int>> c22, int m) {

    int dim = 2 * m;

    vector<vector<int>> result (dim, vector<int>(dim, 0));

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i < m && j < m) {
                result[i][j] = c11[i][j];
            } else if (i < m) {
                result[i][j] = c12[i][j - m];
            } else if (j < m) {
                result[i][j] = c21[i - m][j];
            } else {
                result[i][j] = c22[i - m][j - m];
            }
        }
    }

    return result;




}

vector<vector<int>> strassen(int dim, vector<vector<int>>& upperA, vector<vector<int>>& upperB) {

    int m = dim / 2;

    if (dim == pow(2, k - kPrime)) {

        vector<vector<int>> c (dim, vector<int>(dim, 0));

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    c[i][j] += (upperA[i][k] * upperB[k][j]);
                }
            }
        }

        return c;

    }

    // initialize submatrices for a and b 
    vector<vector<int>> a11 = getSubMat(0, 0, dim, upperA);
    vector<vector<int>> a12 = getSubMat(0, m, dim, upperA);
    vector<vector<int>> a21 = getSubMat(m, 0, dim, upperA);
    vector<vector<int>> a22 = getSubMat(m, m, dim, upperA);

    vector<vector<int>> b11 = getSubMat(0, 0, dim, upperB);
    vector<vector<int>> b12 = getSubMat(0, m, dim, upperB);
    vector<vector<int>> b21 = getSubMat(m, 0, dim, upperB);
    vector<vector<int>> b22 = getSubMat(m, m, dim, upperB);

    // cout << "DEBUG" << endl;

    vector<vector<int>> m1;
    vector<vector<int>> m2;
    vector<vector<int>> m3;
    vector<vector<int>> m4;
    vector<vector<int>> m5;
    vector<vector<int>> m6;
    vector<vector<int>> m7;

    vector<vector<int>> m1x;
    vector<vector<int>> m1y;
    vector<vector<int>> m2x;
    vector<vector<int>> m3x;
    vector<vector<int>> m4x;
    vector<vector<int>> m5x;
    vector<vector<int>> m6x;
    vector<vector<int>> m6y;
    vector<vector<int>> m7x;
    vector<vector<int>> m7y;


    #pragma omp task shared (m1, a11, a22, b11, b22, m, m1x, m1y)
    {

        // M1

        m1x = addMat(a11, a22, m);
        m1y = addMat(b11, b22, m);
        m1 = strassen(m, m1x, m1y);

        m1x.clear();
        m1y.clear();


    }

    #pragma omp task shared (m2, a21, a22, b11, m2x, m)
     {

        // M2
        m2x = addMat(a21, a22, m);
        m2 = strassen(m, m2x, b11);

        m2x.clear();

    }

    #pragma omp task shared (m3, b12, b22, m, a11, m3x)
    {

    // M3
        m3x = subMat(b12, b22, m);
        m3 = strassen(m, a11, m3x);

        m3x.clear();

    }


    #pragma omp task shared (m4, b21, b11, m, a22, m4x)
    {

        // M4
        m4x = subMat(b21, b11, m);
        m4 = strassen(m, a22, m4x);

        m4x.clear();

    }


    #pragma omp task shared (m5, a11, a12, m, m5x, b22)
    {

        // M5
        m5x = addMat(a11, a12, m); 
        m5 = strassen(m, m5x, b22); 

        m5x.clear();


    }

    #pragma omp task shared (m6, a21, a11, m, b11, b12, m6x, m6y)
    {

        // M6
        m6x = subMat(a21, a11, m);
        m6y = addMat(b11, b12, m);
        m6 = strassen(m, m6x, m6y);

        m6x.clear();
        m6x.clear();


    }


    #pragma omp task shared (m7, m7x, m7y, a12, a22, b21, b22, m)
    {

        // M7
        m7x = subMat(a12, a22, m);
        m7y = addMat(b21, b22, m);
        m7 = strassen(m, m7x, m7y);

        m7x.clear();
        m7y.clear();

    }


    #pragma omp taskwait


    a11.clear();
    a12.clear();
    a21.clear();
    a22.clear();
    b11.clear();
    b12.clear();
    b21.clear();
    b22.clear();



    // Combine outputs



   

    // C11
    vector<vector<int>> c11x = addMat(m1, m4, m);
    vector<vector<int>> c11y = addMat(c11x, m7, m);
    vector<vector<int>> c11 = subMat(c11y, m5, m);

    c11x.clear();
    c11y.clear();

    
    // C12
    vector<vector<int>> c12 = addMat(m3, m5, m);
    
    // C21
    vector<vector<int>> c21 = addMat(m2, m4, m);
    
    // C22
    vector<vector<int>> c22x = addMat(m3, m6, m);
    vector<vector<int>> c22y = addMat(c22x, m1, m);
    vector<vector<int>> c22 = subMat(c22y, m2, m);

    c22x.clear();
    c22y.clear();

    m1.clear();
    m2.clear();
    m3.clear();
    m4.clear();
    m5.clear();
    m6.clear();
    m7.clear();
  
    vector<vector<int>> result = combMat(c11, c12, c21, c22, m);

    return result;

}

int main(int argc, char** argv){

    srand(time(0));

    // int k; // Dimension of matrix
    // int kPrime;

    // for (int i = 0; i < argc; i++) {
    //     cout << atoi(argv[i]) << endl;
    // }

    // cout << "Size: " << argc << endl;

    // exit(0);

    int numThr;

    struct timespec start, stop;
    double total_time, time_res;

    if (argc != 4) {
        cout << "All 3 inputs (k, k', # of threads) are needed to compute via Strassen!" << endl;
        cout << "Enter k for Matrix: " << endl;
        cin >> k;

        cout << "Enter k' for Matrix: " << endl;
        cin >> kPrime;

        cout << "Enter number of threads: " << endl;
        cin >> numThr;
    } else {
        k = atoi(argv[1]);
        kPrime = atoi(argv[2]);
        numThr = atoi(argv[3]);
    }


    int n = pow(2, k);

    vector<vector<int>> aUpper = genMatrix(n);
    vector<vector<int>> bUpper = genMatrix(n);
    vector<vector<int>> result;

    clock_gettime(CLOCK_REALTIME, &start);

    omp_set_num_threads(numThr);

    #pragma omp parallel shared (result, n, aUpper, bUpper)
    {
        // cout << "NUM THR: " << omp_get_num_threads() << endl;
        #pragma omp single 
        {

            result = strassen(n, aUpper, bUpper);
        }

    }


    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec) +0.000000001*(stop.tv_nsec-start.tv_nsec);

    cout << "K: " << k << ", K': "<< kPrime << ", Threads: " << numThr << ", Time: " << total_time << endl;


    return 0;
}