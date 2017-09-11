//
//  main.cpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall P. O'Byrnes on 17/19/2.
//  Copyright © 2017 Niall P. O'Byrnes. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <fstream>
#include <iostream>

#include "define_typealias.h"
#include <stdint.h>



/* File should begin with
 VV FLOAT FILE


 256 256
*/


using namespace std;



char inFileName[100];



struct Vec {
    float x,y,z;

    Vec(float px,float py,float pz) : x(px), y(py), z(pz) {}
    Vec(int px, int py, int pz) : x(float(px)), y(float(py)), z(float(pz)) {}
    //__host__ __device__ Vec() {}
    Vec() {}
};




struct VectorData
{
    unsigned indices;
    unsigned prevMsBetweenPoints;
    unsigned lastMpTime;
    float dotSpread;
    float dotBrightness;
    Vec v;
    float rho;
    //float len;
};


std::vector<VectorData> gridVector;






struct SProgramState
{
    int nx;
    int ny;

} ps;





struct Bytes4
{
    uint8_t byte[4];
    void Set(unsigned val){
        //little endian
        byte[0] = val&255;
        byte[1] = (val>>8)&255;
        byte[2] = (val>>16)&255;
        byte[3] = (val>>24)&255;
    }
    Bytes4(unsigned val){ Set(val);}
    Bytes4(int val) { Set(int(val)); }
    Bytes4(float val){
        unsigned *uval=(unsigned*) &val;
        Set(*uval);
    }
};
void WriteBytes4(FILE* file, Bytes4 data)
{
    fwrite(data.byte,4,1,file);
}


bool StrBegEqu(const char* beg, char*str)
{
    while(*beg!=0) {
        if (*beg!=*str) return false;
        beg++;
        str++;
    }
    return true;
}




void ReadLineBeg(FILE* file, char* str, int maxLen)
{
    int i=0;
    while(!feof(file)) {
        int chr = fgetc(file);
        if (chr=='\n') break;
        if (i<maxLen-1)
            str[i++] = char(chr);
    }
    str[i]='\n';
}




float ReadFloat(FILE* file)
{
    float f;
    int ignore = fread(&f,1,4,file);
    return f;
}



//Reads in both text and float file
bool ReadInputFile(const char* inFileName)
{
    int nx;
    int ny;

    FILE* inFile=fopen(inFileName,"rt");
    if (inFile==NULL) return false;

    const int textLineLength=200;
    char str[textLineLength];
    ReadLineBeg(inFile,str,textLineLength);
    if (!StrBegEqu("VV TEXT FILE", str) && !StrBegEqu("VV FLOAT FILE", str)) {
        fprintf(stderr,"Not a VVT of VVF file\n");
        return false;
    }
    bool isVvfFile = StrBegEqu("VV FLOAT FILE", str);
    fclose(inFile);

    inFile=fopen(inFileName, isVvfFile? "rb" : "rt");
    if (inFile==NULL) return false;

    ReadLineBeg(inFile,str,textLineLength); //skip line
    ReadLineBeg(inFile,str,textLineLength); //skip line
    ReadLineBeg(inFile,str,textLineLength); //skip line

    int ignore = fscanf(inFile,"%d%d",&nx, &ny);
    if (nx<60 || ny<60) {
        printf("Input file grid dimensions too small\n");
        return false;
    }
    if (nx*ny>6000*6000) {
        printf("Input file grid dimensions too big\n");
        return false;
    }
    ReadLineBeg(inFile,str,textLineLength); //skip rest of the line

    ps.nx=nx;
    ps.ny=ny;
    gridVector.resize(nx*ny);



    for(int yi=0; yi<ny; yi++)
        for(int xi=0; xi<nx; xi++){
            if (feof(inFile)){
                printf("Unexpected end-of-file in input file\n");
                return false;
            }
            int i = yi*nx + xi;
            float u;
            float x,y,z;
            if (isVvfFile){
                x = ReadFloat(inFile);
                y = ReadFloat(inFile);
                z = ReadFloat(inFile);
            }
            else
                int ignore = fscanf(inFile,"%f%f%f%f",&u,&x,&y,&z);
            gridVector[i].v = Vec( y, -x, z);

            gridVector[i].rho = u;

            gridVector[i].indices= (yi<<16)+xi;
        }

    fclose(inFile);
    return true;
}








//Reads in both text and float file
bool ReadInputFile_FULL(const char* inFileName)
{
    int nx;
    int ny;
    int nz;

    FILE* inFile=fopen(inFileName,"rt");
    if (inFile==NULL) return false;

    const int textLineLength=200;
    char str[textLineLength];
    ReadLineBeg(inFile,str,textLineLength);
    if (!StrBegEqu("VV TEXT FILE", str) && !StrBegEqu("VV FLOAT FILE", str)) {
        fprintf(stderr,"Not a VVT of VVF file\n");
        return false;
    }
    bool isVvfFile = StrBegEqu("VV FLOAT FILE", str);
    fclose(inFile);

    inFile=fopen(inFileName, isVvfFile? "rb" : "rt");
    if (inFile==NULL) return false;

    ReadLineBeg(inFile,str,textLineLength); //skip line
    ReadLineBeg(inFile,str,textLineLength); //skip line
    ReadLineBeg(inFile,str,textLineLength); //skip line

    int ignore = fscanf(inFile,"%d%d%d",&nx, &ny, &nz);
    if (nx<60 || ny<60 || nz<60) {
        printf("Input file grid dimensions too small\n");
        return false;
    }
    if (nx*ny>6000*6000) {
        printf("Input file grid dimensions too big\n");
        return false;
    }
    ReadLineBeg(inFile,str,textLineLength); //skip rest of the line


    gridVector.resize(nx * ny * nz);


    for (int zi=0; zi<nz; zi++)
        for (int yi=0; yi<ny; yi++)
            for (int xi=0; xi<nx; xi++){

                if (feof(inFile)){
                    printf("Unexpected end-of-file in input file\n");
                    return false;
                }

                int pos = (zi * ny * nx) + (yi * nx) + (xi);

                float u;
                float x, y, z;

                x = ReadFloat(inFile);
                y = ReadFloat(inFile);
                z = ReadFloat(inFile);


                gridVector[pos].v = Vec( y, -x, z);

                gridVector[pos].rho = u;

                //gridVector[i].indices= (yi<<16)+xi;
            }
    
    fclose(inFile);
    return true;
}








void SaveVvfFile(char* filename)
{
    FILE* outFile=fopen(filename,"wb");
    if (outFile==NULL) {
        fprintf(stderr,"Error: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(outFile,"VV FLOAT FILE\n");
    fprintf(outFile,"\n");
    fprintf(outFile,"\n");
    
    fprintf(outFile,"%d %d\n",ps.nx, ps.ny);	
    
    for(int i=0; i<ps.nx*ps.ny; i++){						
        Vec v = gridVector[i].v;
        Vec outVec = Vec(-v.y, v.x, v.z);
        WriteBytes4(outFile,Bytes4(outVec.x));
        WriteBytes4(outFile,Bytes4(outVec.y));
        WriteBytes4(outFile,Bytes4(outVec.z));		
    }
    fclose(outFile);
}





//Saves a text file
void SaveVvtFile(char* filename)
{
    FILE* outFile=fopen(filename,"wb");
    if (outFile==NULL) {
        fprintf(stderr,"Error: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(outFile,"VV TEXT FILE\n");
    fprintf(outFile,"\n");
    fprintf(outFile,"\n");

    fprintf(outFile,"%d %d\n",ps.nx, ps.ny);

    for(int i=0; i<ps.nx*ps.ny; i++){
        Vec v = gridVector[i].v;
        Vec outVec = Vec(-v.y, v.x, v.z);

        fprintf(outFile, "%f %f %f", outVec.x, outVec.y, outVec.z);


    }
    fclose(outFile);
}











void merge_native_slice(const char *dirname, t3d startidi, t3d endidi, t3d startidj, t3d endidj, t3d startidk, t3d endidk, int ngx, int snx){



    int sny = snx;

    int nx = snx/ngx;
    int ny = nx;


    std::vector<VectorData> mergedGridVector;

    mergedGridVector.resize(snx * sny);



    for (int idi = startidi; idi <= endidi; idi++){

        for (int idj = startidj; idj <= endidj; idj++){

            for (int idk = startidk; idk <= endidk; idk++){


                char filename[30];
                sprintf(filename, "%s/node.%i.%i.%i.vvf", dirname, idi, idj, idk);

                printf("Opening %s\n", filename);

                ReadInputFile(filename);


                //Now put gridVector into mergedGridVector

                for(int yi=0; yi<ny; yi++){
                    for(int xi=0; xi<nx; xi++){

                        int i = yi*nx + xi;


                        int merged_yi = idj*ny + yi;

                        int merged_xi = idk*ny + xi;


                        int merged_i = ((merged_yi)*snx) + (merged_xi);


                        mergedGridVector[merged_i].v = gridVector[i].v;
                        
                        mergedGridVector[merged_i].rho = gridVector[i].rho;
                        
                        mergedGridVector[merged_i].indices= (merged_yi<<16) + merged_xi;


                    }}

            }}}//end of hijk




    char outputfile[50];


    printf("Saving merged file.\n");

    sprintf(outputfile, "%s/node.vvf", dirname);

    ps.nx = snx;
    ps.ny = sny;

    gridVector = mergedGridVector;

    SaveVvfFile(outputfile);
}



void merge_native_axis(const char *dirname, t3d startidi, t3d endidi, t3d startidj, t3d endidj, t3d startidk, t3d endidk, int ngx, int snx){



    int snz = snx;

    int nx = snx/ngx;
    int ny = nx;



    std::vector<VectorData> mergedGridVector;

    mergedGridVector.resize(snx * snz);


    for (int idi = startidi; idi <= endidi; idi++){

        for (int idj = startidj; idj <= endidj; idj++){

            for (int idk = startidk; idk <= endidk; idk++){


                char filename[30];
                sprintf(filename, "%s/node.%i.%i.%i.vvf", dirname, idi, idj, idk);

                printf("Opening %s\n", filename);

                //Reads grid
                ReadInputFile(filename);


                //Now put gridVector into mergedGridVector

                for(int yi=0; yi<ny; yi++){
                    for(int xi=0; xi<nx; xi++){

                        int i = yi*nx + xi;


                        int merged_yi = idi*ny + yi;

                        int merged_xi = idk*ny + xi;


                        int merged_i = ((merged_yi)*snx) + (merged_xi);


                        mergedGridVector[merged_i].v = gridVector[i].v;
                        
                        mergedGridVector[merged_i].rho = gridVector[i].rho;
                        
                        mergedGridVector[merged_i].indices= (merged_yi<<16) + merged_xi;


                    }}

            }}}//end of hijk




    char outputfile[50];

    sprintf(outputfile, "%s/node.vvf", dirname);

    ps.nx = snx;
    ps.ny = snz;

    gridVector = mergedGridVector;

    SaveVvfFile(outputfile);



}



void merge_native_full(const char *dirname, int ngx, int snx, int xcut, int ycut, int zcut){


    int sny = snx;
    int snz = snx;


    int nx = snx/ngx;
    int ny = nx;
    int nz = nx;


    int ngy = ngx;
    int ngz = ngx;


    printf("ngx=%i nx=%i snx=%i.\nCutting at %i %i %i (two of last 3 should be 0)\n", ngx, nx, snx, xcut, ycut, zcut);



    std::vector<VectorData> mergedGridVector;

    mergedGridVector.resize(snx * sny);


    int idi_start = 0;
    int idj_start = 0;
    int idk_start = 0;

    int idi_end = ngx - 1;
    int idj_end = ngy - 1;
    int idk_end = ngz - 1;

    if (xcut > 0) {idi_start = xcut / nx; idi_end = xcut / nx;}
    else if (ycut > 0) {idj_start = ycut / ny; idj_end = ycut / ny;}
    else if (zcut > 0) {idk_start = zcut / nz; idk_end = zcut / nz;}
    else {}



    int i_start = 0;
    int j_start = 0;
    int k_start = 0;

    int i_end = nx - 1;
    int j_end = ny - 1;
    int k_end = nz - 1;

    if (xcut > 0) {i_start = xcut % nx; i_end = xcut % nx;}
    else if (ycut > 0) {j_start = ycut % nx; j_end = ycut % nx;}
    else if (zcut > 0) {k_start = zcut % nx; k_end = zcut % nx;}
    else {}




    printf(" idi [node_i] %i:%i [ %i:%i]\n idj [node_j] %i:%i [ %i:%i]\n idk [node_k] %i:%i  [ %i:%i] \n", idi_start, idi_end, i_start, i_end, idj_start, idj_end, j_start, j_end, idk_start, idk_end, k_start, k_end);

    for (int idi = idi_start; idi <= idi_end; idi++){

        for (int idj = idj_start; idj <= idj_end; idj++){

            for (int idk = idk_start; idk <= idk_end; idk++){



                char filename[30];
                sprintf(filename, "%s/node.%i.%i.%i.vvf", dirname, idi, idj, idk);

                printf("Opening %s", filename);
                ReadInputFile_FULL(filename);
                printf(", Done.\n", filename);


                //THIS COUNTS THROUGH THE NODE.  THIS IS NOT THE SAME AS SLICE OR AXIS

                for (int node_i = i_start; node_i <= i_end; node_i++){

                    for (int node_j = j_start; node_j <= j_end; node_j++){

                        for (int node_k = k_start; node_k <= k_end; node_k++){


                            //The final image has only 2 dimensions
                            int imgu = 0;
                            int imgv = 0;

                            if (xcut > 0) {imgu = idj * ny + node_j; imgv = idk * nz + node_k;}
                            if (ycut > 0) {imgu = idi * nx + node_i; imgv = idk * nz + node_k;}
                            if (zcut > 0) {imgu = idi * nx + node_i; imgv = idj * ny + node_j;}


                            int merged_pos = (imgu * snx) + imgv;


                            int node_pos = (node_i * ny * nx) + (node_j * nx) + (node_k);

                            mergedGridVector[merged_pos].v = gridVector[node_pos].v;

                            mergedGridVector[merged_pos].rho = gridVector[node_pos].rho;


            }}}//end of node_ijk

            }}}//end of idijk
    
    
    
    char outputfile[50];

    sprintf(outputfile, "%s/node_%i_%i_%i.vvf", dirname, xcut, ycut, zcut);

    ps.nx = snx;
    ps.ny = snz;

    gridVector = mergedGridVector;

    SaveVvfFile(outputfile);


}





int main(int argc, char **argv){


    printf("merge-tool dirname ngx snx [x_cut y_cut z_cut]\n");


    string dirname (argv[1]);
    string slice ("slice");
    string axis ("axis");
    string full ("full");

    int ngx = atoi(argv[2]);

    int snx = atoi(argv[3]);



    if (dirname.find(slice) != std::string::npos) {
        std::cout << "Merging Slice" << '\n';
        merge_native_slice(argv[1], 1, 1, 0, ngx-1, 0, ngx-1, ngx, snx);


        printf("not getting to here\n");
    }
    else if (dirname.find(axis) != std::string::npos) {
        std::cout << "Merging Axis" << '\n';
        merge_native_axis(argv[1], 0, ngx-1, 1, 1, 0, ngx-1, ngx, snx);

    }
    else if (dirname.find(full) != std::string::npos) {
        std::cout << "Merging Full" << '\n';

        int xcut = atoi(argv[4]);
        int ycut = atoi(argv[5]);
        int zcut = atoi(argv[6]);

        merge_native_full(argv[1], ngx, snx, xcut, ycut, zcut);

    }

    printf("end of prg\n");

    return 0;

}
