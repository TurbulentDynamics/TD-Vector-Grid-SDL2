#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <SDL.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "renderer.h"
#include "SipYAML.hpp"

float PI=3.1415926535897932384f;

char inFileName[10000];

const float cBrightnessMultiplier = 45;
const float cLengthMultiplier = 0.1f;
const unsigned cMaxDotsPerLine = 30;

const float cBrightnessMin = 0.2f;
const float cBrightnessMax = 900.0f;
const float cBrightnessStep = 0.003f;

const float cLengthMin = 2.0f;
const float cLengthMax = 900.0f;
const float cLengthStep = 0.003f;

const int cudaMaxPoints = 24*1000*1000;
float gridSize = 1;

float rotLRInit = 0;
float rotUDInit = -20;
float distInit = 150;
float centerXInit = 0;
float centerYInit = 0;
float centerZInit = 0;
float brightnessInit = 100.0f;
float lengthInit = 100.0f;
float maxLength = 0;
float colorScale = 1.0f;
int initialDotDensity = -2;
int timeToSimulate = 0;

char *dumpFilename = NULL;
bool showParams = false;
bool offScreen = false;
bool useColor = false;
bool useSpeed = false;
bool useInPlane = false;
bool useOrtho = false;
bool showHeatmap = false;
bool showVectors = true;
int displayMode = 0;

const char* programName = "VectorViz v1.00";

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>	
	#undef min
	#undef max
	//#define USE_DEFAULT_FILE
#endif

#ifdef USE_DEFAULT_FILE
	//const char* inFileNameDefault="flowyz_nx_01536_0012000_vect.vvf";
	const char* inFileNameDefault="flowyz0005030.vvf";
#endif

#if !defined(_DEBUG) && !defined(NO_FULLSCREEN)
	//#define FULLSCREEN
#endif

#if defined(FULLSCREEN)
	int screenW = 0;
	int screenH = 0;
#else	
	int screenW = 1024;
	int screenH = 768;
#endif

SDL_Window* display;
SDL_Surface* screen;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	Uint32 rmask = 0xff000000;
	Uint32 gmask = 0x00ff0000;
	Uint32 bmask = 0x0000ff00;
	Uint32 amask = 0x000000ff;
#else
	Uint32 rmask = 0x000000ff;
	Uint32 gmask = 0x0000ff00;
	Uint32 bmask = 0x00ff0000;
	Uint32 amask = 0xff000000;
#endif

const uint8_t* GetGlyphBmp(int c);
int GetGlyphPitch();
int GetGlyphWidth();
int GetGlyphHeight(); 

unsigned kiss_z     = 123456789;  // 1 - billion
unsigned kiss_w     = 378295763;  // 1 - billion
unsigned kiss_jsr   = 294827495;  // 1 - RNG_MAX
unsigned kiss_jcong = 495749385;  // 0 - RNG_MAX

unsigned RngUns()
{
	//MWC x2
	kiss_z=36969*(kiss_z&65535)+(kiss_z>>16);
	kiss_w=18000*(kiss_w&65535)+(kiss_w>>16);

	//SHR3
	kiss_jsr^=(kiss_jsr<<13); 
	kiss_jsr^=(kiss_jsr>>17); 
	kiss_jsr^=(kiss_jsr<<5);

	//linear congruential
	kiss_jcong=69069*kiss_jcong+1234567;	
	return (((kiss_z<<16)+kiss_w)^kiss_jcong)+kiss_jsr;
}

const unsigned RNG_MAX = 4294967295; // 2^32 - 1

void UnrollRng()
{
	for (int i=0; i<1000; i++)
		RngUns();
}
// Random number form interval <0, 1>
double Rng01()
{        
	return (RngUns()+0.5) / (RNG_MAX+1.0);
}

class TIMER
{	
	//unsigned resetTime;
	//time of resetting
	unsigned resetTime;
	public:
	TIMER() {}	
	// void  Reset();
	// Resets timer to 0.
	void  Reset()
	{	resetTime=(unsigned)SDL_GetTicks();  }	
	// unsigned E();
	// Returns time in miliseconds that has 
	// passed from creation or from last Reset().
	unsigned Get()
	{	return (unsigned)SDL_GetTicks()-resetTime;	}
	void Subtract(int time)
	{	resetTime+=time;	}
};

inline Vec VecAdd(Vec a, Vec b) {	
	return Vec(a.x+b.x, a.y+b.y, a.z+b.z);	
}
inline Vec VecSub(Vec a, Vec b) {
	return Vec(a.x-b.x, a.y-b.y, a.z-b.z);	
}
inline  float DotProduct(Vec A, Vec B) { 
	return A.x*B.x + A.y*B.y + A.z*B.z;
		}
inline float VecLen(Vec a) { 
	return sqrt(DotProduct(a,a));
	}
inline Vec VecMul(Vec v,float t) { 
	v.x*=t;
	v.y*=t;
	v.z*=t;
	return v;	
	}   
inline Vec VecUnit(Vec a) { 
	return VecMul(a, 1.0f/VecLen(a));
	}
inline Vec VecCross(Vec a, Vec b) {
	return Vec(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.z);
}

Vec RotUD(Vec v, float fi) //elevation
{
	Vec r;
	r.x = v.x;
	r.z = cosf(fi)*v.z - sinf(fi) *v.y;
	r.y = sinf(fi)*v.z + cosf(fi) *v.y;
	return r;
}

Vec RotLR(Vec v, float fi) // azimuth
{
	Vec r;
	r.y = v.y;
	r.x = cosf(fi)*v.x - sinf(fi) *v.z;        
	r.z = sinf(fi)*v.x + cosf(fi) *v.z;        
	return r;
}

Vec SphericalRotation(Vec v, float elevation, float azimuth)
{
	return RotLR( RotUD(v,elevation), azimuth);
}

struct MovingPoint
{
	Vec s;               // (position) start of line/vector in the grid
	float brightness;	
	Vec v;               // vector from start to the end of the line
	float offset;        // in range 0-1, describes the current place of the moving point
};


std::vector<MovingPoint> createdMp;
std::vector<float> separatedMp;
std::vector<SMpBufDesc> mpBufDescCuda;
std::vector<SQuadBufDesc> quadBufDescCuda;
std::vector<float> quadData;


struct SCudaMemory
{	
	float* mpData;
	SMpBufDesc* mpBufDesc;
	unsigned* intensityRaster;
	SQuadBufDesc *quadBufDesc;
	float *quadData;
} cudaMem;

struct CameraArrangement
{
	float dist;
	float rotUD;
	float rotLR;
	Vec centerTranslation;                
} cameraArrange;

Camera CreateCamera(CameraArrangement pk)
{
	float azimuth = pk.rotLR/180*PI;
	float elevation = pk.rotUD/180*PI;
	
	Camera k;        
	k.dir = Vec(0,0,1);        
	k.xd = Vec(1,0,0);
	k.yd = Vec(0,-1,0);
	k.screenDist = screenW+screenW/2.0f;	
	k.dir = SphericalRotation( k.dir, elevation, azimuth) ;                
	k.xd     = SphericalRotation( k.xd, elevation, azimuth );
	k.yd   = SphericalRotation( k.yd, elevation, azimuth );
	k.upLeftCornerTrans = VecAdd( VecMul(k.xd,screenW/2.0f-0.5f), VecMul(k.yd,screenH/2.0f-0.5f) );	
	
	k.eye = SphericalRotation( Vec(0.0f,0.0f,-pk.dist), elevation, azimuth);
	k.eye = VecAdd(k.eye, pk.centerTranslation);
	
	return k;
}

struct IntensityBuffer
{
	unsigned* pix;
	int sizeX;
	int sizeY;
} intRaster;


int GammaLinearToColorSRGB(double col)
{    
	if (col<=0.00304)
		return int(col*12.92*255);
	
	double up=1.055*pow(col,1/2.4)-0.055;
	return int(up*255);
}

std::vector<int> gammaPre;
static const unsigned gammaPreSize=16000;

void SetupGammaPre(){
	double step=1.0/gammaPreSize;
	for (double i=0;i<gammaPreSize;i++) {
		double midVal=i/gammaPreSize+step/2;
		gammaPre.push_back( GammaLinearToColorSRGB(midVal) );
		}
	gammaPre.push_back( 255 );
	}

struct SSys
{
	bool isActive;
	bool isQuit;
	bool isMouseLeftPressed;
	bool isMouseRightPressed;
	bool isKeyLeftPressed;
	bool isKeyRightPressed;
	bool isKeyUpPressed;
	bool isKeyDownPressed;
	bool isKeyPgUpPressed;	
	bool isKeyPgDnPressed;	
	bool isKeyHomePressed;	
	bool isKeyEndPressed;
	bool isKeyPadUp;	
	bool isKeyPadDown;	
	bool isKeyPadLeft;	
	bool isKeyPadRight;		
};

struct SProgramState
{	
	SSys sys;				
	
	int pointCntExp;
	float exposure;
	float totalBrightness;
	float length;
	bool isAutoAdjust;	
	int nx;
	int ny;	
	int nz;
	
	bool isScreenshotRequest;
	bool isPaused;
	unsigned curTime;
	TIMER programTime;
	TIMER pressMouseRightTime;
	TIMER pressMouseLeftTime;	
	TIMER pressKeyLeftTime;	
	TIMER pressKeyRightTime;
	TIMER pressKeyUpTime;
	TIMER pressKeyDownTime;	
	TIMER pressKeyPgUpTime;
	TIMER pressKeyPgDnTime;
	TIMER pressKeyHomeTime;
	TIMER pressKeyEndTime;
	TIMER pressKeyPadDownTime;	
	TIMER pressKeyPadUpTime;
	TIMER pressKeyPadLeftTime;
	TIMER pressKeyPadRightTime;	
}ps;

struct VectorData
{		
	uint64_t indices;
	unsigned prevMsBetweenPoints;
	unsigned lastMpTime;
	float dotSpread;		
	float dotBrightness;		
	Vec v;
	//float len;		
};

std::vector<VectorData> gridVector;

VectorData CreateGridVector(Vec inVec, uint64_t indices, float maxDotSpread)
{
	VectorData gridVec;
	gridVec.v = inVec;
	float length = VecLen(inVec);
	float dotSpread = 1/length;
	if (dotSpread>maxDotSpread) dotSpread=maxDotSpread;
	gridVec.dotSpread = dotSpread;
	//gridVec.len = length;	
	gridVec.dotBrightness = length*length;		
	gridVec.prevMsBetweenPoints	= 300;
	gridVec.lastMpTime =  0-unsigned(Rng01()*300);	
	gridVec.indices = indices;	
	return gridVec;
}


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

void PermutateGridVectors()
{
	for(int i=0; i<gridVector.size(); i++) {
		VectorData temp = gridVector[i];
		int rndI = unsigned(Rng01()*(gridVector.size()-i))+i;
		gridVector[i] = gridVector[rndI];
		gridVector[rndI] = temp;
		}
}

float ReadFloat(FILE* file)
{
	float f;
	int ignore = fread(&f,1,4,file);			
	return f;
}

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
	if (nx*ny>6000*6000) {
		printf("Input file grid dimensions too big\n");
		return false;
		}
	ReadLineBeg(inFile,str,textLineLength); //skip rest of the line
	
	ps.nx=1;
	ps.ny=ny;
	ps.nz=nx;
	gridVector.resize(nx*ny);
	if (showHeatmap) quadData.resize(nx * ny);
	
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
			gridVector[i].v = Vec( x, y, z);
			gridVector[i].indices= (xi) + ((uint64_t)yi<<21);
			auto length = VecLen(gridVector[i].v);
			maxLength = std::max(maxLength, length);
			if (showHeatmap) quadData[i] = length;
			}	
	
	if (showHeatmap)
	{
		quadBufDescCuda.resize(1);
		auto &desc = quadBufDescCuda.front();
		desc.gridWidth = nx;
		desc.gridHeight = ny;
		desc.vertices[0] = Vec(0.f, nx *  0.5f, ny * -0.5f);
		desc.vertices[1] = Vec(0.f, nx * -0.5f, ny * -0.5f);
		desc.vertices[2] = Vec(0.f, nx *  0.5f, ny *  0.5f);
		desc.vertices[3] = Vec(0.f, nx * -0.5f, ny *  0.5f);
		desc.dataOffset = 0;
	}

	fclose(inFile);
	return true;
}

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
//  if (nx<60 || ny<60 || nz<60) {
//    printf("Input file grid dimensions too small\n");
//    return false;
//  }
	if (nx*ny>6000*6000) {
		printf("Input file grid dimensions too big\n");
		return false;
	}
	ReadLineBeg(inFile,str,textLineLength); //skip rest of the line

	gridVector.resize(nx * ny * nz);
	if (showHeatmap) quadData.resize(nx * ny * nz);

	for (int zi=0; zi<nz; zi++)
		for (int yi=0; yi<ny; yi++)
			for (int xi=0; xi<nx; xi++)
			{
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


				gridVector[pos].v = Vec( x, y, z);
				auto length = VecLen(gridVector[pos].v);
				maxLength = std::max(maxLength, length);
				if (showHeatmap) quadData[pos] = length;

				//gridVector[pos].indices= (yi<<16)+xi;
			}

	fclose(inFile);
	return true;
}

void SaveVvfFile(char* filename, std::vector<VectorData> const &grid, int nx, int ny, int nz)
{
	FILE* outFile=fopen(filename,"wb");
	if (outFile==NULL) {
		fprintf(stderr,"Error: Cannot open %s for writing\n", filename);
		return;
		}
	
	fprintf(outFile,"VV FLOAT FILE\n");
	fprintf(outFile,"\n");
	fprintf(outFile,"\n");
		
	if (nz == 1)
	{
		fprintf(outFile,"%d %d\n", nx, ny);
	}
	else
	{
		fprintf(outFile,"%d %d %d\n", nx, ny, nz);
	}
	
	for(int i=0; i<nx*ny*nz; i++){
		Vec v = grid[i].v;
		Vec outVec = Vec(-v.y, v.x, v.z);
		WriteBytes4(outFile,Bytes4(outVec.x));
		WriteBytes4(outFile,Bytes4(outVec.y));
		WriteBytes4(outFile,Bytes4(outVec.z));
		}
	fclose(outFile);
}

bool merge_native_slice(const char *dirname, char const *nameRoot, int ngx, int ngy, int ngz, int snx, int sny, int snz, char *outFilename){


    std::string str (dirname);
    std::string str2 ("cut");
    std::size_t found = str.find(str2);
    std::string str_cuti = str.substr (found + 4);

    int cuti = std::stoi(str_cuti);

    int nx = snx / ngx;
    int ny = sny / ngy;


    int startidi = cuti / nx;
    int endidi = startidi;



    int startidj = 0;
    int endidj = ngy - 1;
    int startidk = 0;
    int endidk = ngz - 1;




	std::vector<VectorData> mergedGridVector;
	std::vector<float> mergedQuadData;
	mergedGridVector.resize(snx * sny);
	if (showHeatmap) mergedQuadData.resize(snx * sny);

	for (int idi = startidi; idi <= endidi; idi++)
	{
		for (int idj = startidj; idj <= endidj; idj++)
		{
			for (int idk = startidk; idk <= endidk; idk++)
			{
				char filename[300];
				sprintf(filename, "%s/%s.%i.%i.%i.vvf", dirname, nameRoot, idi, idj, idk);

				printf("Opening %s\n", filename);

				if (!ReadInputFile(filename)) return false;

				for(int yi=0; yi<ny; yi++)
				{
					for(int xi=0; xi<nx; xi++)
					{
						int i = yi*nx + xi;

						int merged_yi = idj*ny + yi;
						int merged_xi = idk*ny + xi;

						int merged_i = ((merged_yi)*snx) + (merged_xi);

						mergedGridVector[merged_i].v = gridVector[i].v;
						mergedGridVector[merged_i].indices = (merged_yi) + ((uint64_t)merged_xi<<21);
						if (showHeatmap) mergedQuadData[merged_i] = quadData[i];
					}
				}
			}
		}
	}

	ps.nx = 1;
	ps.ny = sny;
	ps.nz = snx;

	gridVector = mergedGridVector;
	if (showHeatmap)
	{
		quadData = mergedQuadData;
		quadBufDescCuda.resize(1);
		auto &desc = quadBufDescCuda.front();
		desc.gridWidth = snx;
		desc.gridHeight = sny;
		desc.vertices[0] = Vec(0.f, snx * -0.5f, sny *  0.5f);
		desc.vertices[1] = Vec(0.f, snx * -0.5f, sny * -0.5f);
		desc.vertices[2] = Vec(0.f, snx *  0.5f, sny *  0.5f);
		desc.vertices[3] = Vec(0.f, snx *  0.5f, sny * -0.5f);
		desc.dataOffset = 0;
	}

	if (outFilename)
	{
		SaveVvfFile(outFilename, gridVector, snx, sny, 1);
	}

	return true;
}

bool merge_native_axis(const char *dirname, char const *nameRoot, int ngx, int ngy, int ngz, int snx, int sny, int snz, char *outFilename)
{

    std::string str (dirname);
    std::string str2 ("cut");
    std::size_t found = str.find(str2);
    std::string str_cutj = str.substr (found + 4);

    int cutj = std::stoi(str_cutj);

    int nx = snx / ngx;
    int ny = sny / ngy;

    int startidj = cutj / ny;
    int endidj = startidj;

    int startidi = 0;
    int endidi = ngx - 1;
    int startidk = 0;
    int endidk = ngz - 1;




	std::vector<VectorData> mergedGridVector;
	std::vector<float> mergedQuadData;
	mergedGridVector.resize(snx * snz);
	if (showHeatmap) mergedQuadData.resize(snx * sny);

	for (int idi = startidi; idi <= endidi; idi++)
	{
		for (int idj = startidj; idj <= endidj; idj++)
		{
			for (int idk = startidk; idk <= endidk; idk++)
			{
				char filename[300];
				sprintf(filename, "%s/%s.%i.%i.%i.vvf", dirname, nameRoot, idi, idj, idk);

				printf("Opening %s\n", filename);

				if (!ReadInputFile(filename)) return false;

				for(int yi=0; yi<ny; yi++)
				{
					for(int xi=0; xi<nx; xi++)
					{
						int i = yi*nx + xi;

						int merged_yi = idi*ny + yi;
						int merged_xi = idk*ny + xi;

						int merged_i = ((merged_yi)*snx) + (merged_xi);

						mergedGridVector[merged_i].v = gridVector[i].v;
						mergedGridVector[merged_i].indices= ((uint64_t)merged_yi<<42) + ((uint64_t)merged_xi<<21);
						if (showHeatmap) mergedQuadData[merged_i] = quadData[i];
					}
				}
			}
		}
	}

	ps.nx = snx;
	ps.ny = snz;
	ps.nz = 1;

	gridVector = mergedGridVector;

	if (showHeatmap)
	{
		quadData = mergedQuadData;
		quadBufDescCuda.resize(1);
		auto &desc = quadBufDescCuda.front();
		desc.gridWidth = snx;
		desc.gridHeight = sny;
		desc.vertices[0] = Vec(sny *  0.5f, snx * -0.5f, 0.f);
		desc.vertices[1] = Vec(sny * -0.5f, snx * -0.5f, 0.f);
		desc.vertices[2] = Vec(sny *  0.5f, snx *  0.5f, 0.f);
		desc.vertices[3] = Vec(sny * -0.5f, snx *  0.5f, 0.f);
		desc.dataOffset = 0;
	}

	if (outFilename)
	{
		SaveVvfFile(outFilename, gridVector, snx, sny, 1);
	}

	return true;
}

bool merge_native_full(const char *dirname, char const *nameRoot, int ngx, int ngy, int ngz, int snx, int sny, int snz)
{
	int nx = snx/ngx;
	int ny = sny/ngy;
	int nz = snz/ngz;

	std::vector<VectorData> mergedGridVector;
	mergedGridVector.resize(snx * sny * snz);

	for (int idi = 0; idi < ngx; idi++)
	{
		for (int idj = 0; idj < ngy; idj++)
		{
			for (int idk = 0; idk < ngz; idk++)
			{
				char filename[300];
				sprintf(filename, "%s/%s.%i.%i.%i.vvf", dirname, nameRoot, idi, idj, idk);

				printf("Opening %s", filename);
				if (!ReadInputFile_FULL(filename)) return false;
				printf(", Done.\n");

				//THIS COUNTS THROUGH THE NODE.  THIS IS NOT THE SAME AS SLICE OR AXIS
				for (int node_i = 0; node_i < nx; node_i++)
				{
					for (int node_j = 0; node_j < ny; node_j++)
					{
						for (int node_k = 0; node_k < nz; node_k++)
						{
							int imgu = idi * nx + node_i;
							int imgv = idj * ny + node_j;
							int imgw = idk * nz + node_k;

							int merged_pos = ((imgw * sny) + imgv) * snx + imgu;

							int node_pos = (node_i * ny * nx) + (node_j * nx) + (node_k);

							mergedGridVector[merged_pos].v = gridVector[node_pos].v;
							mergedGridVector[merged_pos].indices = ((uint64_t)imgu << 42) + ((uint64_t)imgv << 21) + imgw;
						}
					}
				}
			}
		}
	}

	ps.nx = snx;
	ps.ny = sny;
	ps.nz = snz;

	gridVector = mergedGridVector;

	return true;
}

void RasterLine(std::vector<int2> &points, int x0, int y0, int x1, int y1)
{
	int dx =  abs(x1-x0), sx = x0<x1 ? 1 : -1;
	int dy = -abs(y1-y0), sy = y0<y1 ? 1 : -1;
	int err = dx+dy, e2;

	for(;;)
	{
		int2 point = { x0, y0 };
		points.push_back(point);
		if (x0==x1 && y0==y1) break;
		e2 = 2*err;
		if (e2 > dy) { err += dy; x0 += sx; }
		if (e2 < dx) { err += dx; y0 += sy; }
	}
}

bool ReadMergeInput(char const *dir, char const *nameRoot, int ngx, int ngy, int ngz, int snx, int sny, int snz, int plane, std::vector<int> &position, int axis, std::vector<int> &phi, char *outFilename, char *outNameRoot)
{

	if (strstr(dir, "slice"))
	{
     	return merge_native_slice(dir, nameRoot, ngx, ngy, ngz, snx, sny, snz, outFilename);
	}

	if (strstr(dir, "axis"))
	{
		return merge_native_axis(dir, nameRoot, ngx, ngy, ngz, snx, sny, snz, outFilename);
	}

	if (strstr(dir, "full"))
	{
		struct Slice
		{
			std::vector<int2> points;
			int h;
			int axis;
			Vec normal;
			Vec xAxis;
			bool isAngle;
			int param;
			int begin;
			int size;
		};

		std::vector<Slice> slices;
		slices.reserve(phi.size() + position.size());

		int bx;
		int by;
		int h;
		switch (axis)
		{
			case 0: bx = sny; by = snz; h = snx; break;
			case 1: bx = snx; by = snz; h = sny; break;
			case 2: bx = snx; by = sny; h = snz; break;
			default:
				fprintf(stdout, "Please specify output/angle/axis using x, y or z\n");
				return false;
		}

		for (auto angle: phi)
		{
			slices.emplace_back();
			auto &slice = slices.back();
			auto cx = (bx - 1) * 0.5f;
			auto cy = (by - 1) * 0.5f;
			auto rad = angle / 180.f * PI;
			auto s = sin(rad);
			auto c = cos(rad);
			auto normal = Vec((axis == 2) ? 1 : 0, (axis == 0) ? 1 : 0, (axis == 1) ? 1 : 0);
			slice.normal = normal;
			auto s2 = s;
			auto c2 = c;
			if (axis == 0)
			{
				slice.normal.y = c;
				slice.normal.z = -s;
			}
			else if (axis == 1)
			{
				slice.normal.x = s;
				slice.normal.z = -c;
			}
			else if (axis == 2)
			{
				slice.normal.x = c;
				slice.normal.y = -s;
			}

			if (fabs(s) / cy > fabs(c) / cx)
			{
				c = c / s * cy;
				s = cy;
				c2 = c2 / s2 * by;
				s2 = by;
			}
			else
			{
				s = s / c * cx;
				c = cx;
				s2 = s2 / c2 * bx;
				c2 = bx;
			}

			slice.xAxis = Vec(0, 0, 0);
			if (axis == 0)
			{
				slice.xAxis.y = s2;
				slice.xAxis.z = c2;
				RasterLine(slice.points, cx - s, cy - c, cx + s, cy + c);
			}
			else if (axis == 1)
			{
				slice.xAxis.x = c2;
				slice.xAxis.z = s2;
				RasterLine(slice.points, cx - c, cy - s, cx + c, cy + s);
			}
			else if (axis == 2)
			{
				slice.xAxis.x = s2;
				slice.xAxis.y = c2;
				RasterLine(slice.points, cx - s, cy - c, cx + s, cy + c);
			}

			slice.h = h;
			slice.axis = axis;
			slice.isAngle = true;
			slice.param = angle;
		}

		if (plane < 0 || plane > 2)
		{
			fprintf(stdout, "Please specify output/ortho/plane using x, y or z\n");
			return false;
		}

		for (auto pos: position)
		{
			slices.emplace_back();
			auto &slice = slices.back();
			int2 beg;
			int2 end;
			switch (plane)
			{
				case 0: beg = make_int2(pos,   0); end = make_int2(pos + 1,     sny); h = snz; break;
				case 1: beg = make_int2(pos,   0); end = make_int2(pos + 1,     snz); h = snx; break;
				case 2: beg = make_int2(  0, pos); end = make_int2(    snx, pos + 1); h = sny; break;
			}

			for (int2 point = beg; point.x < end.x; ++point.x)
			{
				for (point.y = beg.y; point.y < end.y; ++point.y)
				{
					slice.points.push_back(point);
				}
			}

			slice.h = h;
			slice.axis = (plane + 2) % 3;
			slice.xAxis = Vec((plane == 2) * snx, (plane == 0) * sny, (plane == 1) * snz);
			slice.normal = Vec((plane == 0) ? 1 : 0, (plane == 1) ? 1 : 0, (plane == 2) ? 1 : 0);
			slice.isAngle = false;
			slice.param = pos;
		}

		int gridSize = 0;
		for (auto &slice: slices)
		{
			slice.begin = gridSize;
			slice.size = slice.points.size() * slice.h;
			gridSize += slice.size;
		}

		std::vector<VectorData> mergedGridVector(gridSize);
		std::vector<float> mergedQuadData(showHeatmap ? gridSize : 0);

		int nx = snx/ngx;
		int ny = sny/ngy;
		int nz = snz/ngz;

		for (int idi = 0; idi < ngx; idi++)
		{
			for (int idj = 0; idj < ngy; idj++)
			{
				for (int idk = 0; idk < ngz; idk++)
				{
					bool fileLoaded = false;
					auto nodeBegin = make_int3(
						idi * nx,
						idj * ny,
						idk * nz);
					auto nodeEnd = make_int3(
						nodeBegin.x + nx,
						nodeBegin.y + ny,
						nodeBegin.z + nz);

					for (auto &slice: slices)
					{
						int x;
						int y;
						int z;
						int *pX;
						int *pY;
						int *pZ;
						switch (slice.axis)
						{
							case 0: pX = &y; pY = &z; pZ = &x; break;
							case 1: pX = &x; pY = &z; pZ = &y; break;
							case 2: pX = &x; pY = &y; pZ = &z; break;
						}
						for (*pZ = 0; *pZ < h; ++*pZ)
						{
							for (int iPoint = 0; iPoint < slice.points.size(); ++iPoint)
							{
								auto &point = slice.points[iPoint];
								*pX = point.x;
								*pY = point.y;
								if (nodeBegin.x <= x && nodeBegin.y <= y && nodeBegin.z <= z &&
									x < nodeEnd.x && y < nodeEnd.y && z < nodeEnd.z)
								{
									if (!fileLoaded)
									{
										char filename[300];
										sprintf(filename, "%s/%s.%i.%i.%i.vvf", dir, nameRoot, idi, idj, idk);
										printf("Opening %s", filename);
										if (!ReadInputFile_FULL(filename)) return false;
										printf(", Done.\n");
										fileLoaded = true;
									}

									auto mergedIndex = slice.begin + (*pZ * slice.points.size()) + iPoint;
									auto nodePos = make_int3(
										x - nodeBegin.x,
										y - nodeBegin.y,
										z - nodeBegin.z);
									auto nodeIndex =  ((nodePos.x * ny) + nodePos.y) * nx + nodePos.z;
									if (useInPlane)
									{
										Vec projPos = gridVector[nodeIndex].v;
										auto d = DotProduct(projPos, slice.normal);
										projPos = Vec(projPos.x - d * slice.normal.x, projPos.y - d * slice.normal.y, projPos.z - d * slice.normal.z);
										gridVector[nodeIndex].v = projPos;
									}
									mergedGridVector[mergedIndex] = gridVector[nodeIndex];
									mergedGridVector[mergedIndex].indices = ((uint64_t)x << 42) + ((uint64_t)y << 21) + z;
									if (showHeatmap) mergedQuadData[mergedIndex] = quadData[nodeIndex];
								}
							}
						}
					}
				}
			}
		}

		if (outNameRoot)
		{
			for (auto &slice: slices)
			{
				char buffer[300];
				if (slice.isAngle)
				{
					sprintf(buffer, "%s_deg_%c_%d.vvf", outNameRoot, "xyz"[axis], slice.param);
				}
				else
				{
					sprintf(buffer, "%s_cut_%c_%d.vvf", outNameRoot, "xyz"[plane], slice.param);
				}

				SaveVvfFile(buffer, std::vector<VectorData>(mergedGridVector.begin() + slice.begin, mergedGridVector.begin() + slice.begin + slice.size), slice.h, slice.points.size(), 1);
			}
		}

		ps.nx = snx;
		ps.ny = sny;
		ps.nz = snz;

		gridVector = mergedGridVector;

		if (showHeatmap)
		{
			quadData = mergedQuadData;
			quadBufDescCuda.resize(0);
			for (auto &slice: slices)
			{
				quadBufDescCuda.emplace_back();
				auto &desc = quadBufDescCuda.back();
				desc.gridWidth = slice.points.size();
				desc.gridHeight = slice.h;
				Vec yAxis(slice.axis == 0, slice.axis == 1, slice.axis == 2);
				Vec zAxisSize = slice.isAngle
					? Vec(0, 0, 0)
					: Vec(
						slice.param - (snx - 1) * 0.5f,
						slice.param - (sny - 1) * 0.5f,
						slice.param - (snz - 1) * 0.5f);
				Vec zAxis = Vec(slice.normal.x * zAxisSize.x, slice.normal.y * zAxisSize.y, slice.normal.z * zAxisSize.z);
				desc.vertices[0] = VecAdd(VecAdd(VecMul(slice.xAxis, -0.5f), VecMul(yAxis, slice.h *  0.5f)), zAxis);
				desc.vertices[1] = VecAdd(VecAdd(VecMul(slice.xAxis, -0.5f), VecMul(yAxis, slice.h * -0.5f)), zAxis);
				desc.vertices[2] = VecAdd(VecAdd(VecMul(slice.xAxis,  0.5f), VecMul(yAxis, slice.h *  0.5f)), zAxis);
				desc.vertices[3] = VecAdd(VecAdd(VecMul(slice.xAxis,  0.5f), VecMul(yAxis, slice.h * -0.5f)), zAxis);
				desc.dataOffset = slice.begin;
			}
		}

		return true;
	}

	return false;
}

void AnimationStep(unsigned deltaT_ms)
{
	//deltaT_ms/=4;
	/*const double deltaT = deltaT_ms /1000.0;
	const double speed = 0.3;
	ps.offsetAdd += int(65536*speed*deltaT);
	ps.offsetAdd %=65536;		*/
	ps.curTime += deltaT_ms;
}

void PutPixel(int x, int y, unsigned color)
{
	Uint32 pitch = (screen->pitch)/4;
	Uint32* ptr= (Uint32*) screen->pixels;
	
	if (x<0 || x>=screenW) return;
	if (y<0 || y>=screenH) return;
	
	ptr[x+y*pitch]=color;
}

void DrawCharacter(int c, int x, int y)
{
	const uint8_t* bmp = GetGlyphBmp(c);
	int fontH=GetGlyphHeight();
	int fontW=GetGlyphWidth();
	int fontPitch=GetGlyphPitch();
	
	Uint32 pitch = (screen->pitch)/4;
	Uint32* ptr= (Uint32*) screen->pixels;
	
	for (int j=0; j<fontH; j++)
		for (int i=0; i<fontW; i++) {
			int destX=x+i;
			int destY=y+j;
			if (destX<0 || destX>=screenW) continue;
			if (destY<0 || destY>=screenH) continue;
	
			unsigned alpha = bmp[i+j*fontPitch];
			alpha = (13*alpha)>>4;
			unsigned color = (alpha<<16)+(alpha<<8)+alpha;
			ptr[destX+destY*pitch] = color;			
			}	
}

void DrawString(int x, int y, const char* format, ...)
{
		const int maxSize = 1024*2;
		char buf[maxSize];
		va_list argptr;
		va_start(argptr, format);
		vsnprintf (buf, maxSize, format, argptr);
		va_end(argptr);

		int len = strlen(buf);
		for (int i=0;i<len;i++)
		DrawCharacter(buf[i],x+i*GetGlyphWidth(),y);
}

void ClearIntensityRaster()
{
	for (int i=0;i<intRaster.sizeX*intRaster.sizeY*3;i++)
		intRaster.pix[i]=0;
}

void ClearIntensityRasterCuda()
{
	cudaMemset(cudaMem.intensityRaster,0,screenW*screenH*sizeof(unsigned)*3 );
}

void RenderIntensityRaster()
{
	Uint32 pitch = (screen->pitch)/4;
	Uint32* ptr= (Uint32*) screen->pixels;
	
	for (int j=0; j<intRaster.sizeY;j++)
		for (int i=0;i<intRaster.sizeX;i++){
			int srcIndex = (i + j*intRaster.sizeX) * 3;
			unsigned intensityR = intRaster.pix[srcIndex];
			unsigned intensityG = intRaster.pix[srcIndex + 1];
			unsigned intensityB = intRaster.pix[srcIndex + 2];
	
			ptr[i+j*pitch]=
				(gammaPre[std::min(intensityR, gammaPreSize)] * (rmask & 0x01010101)) +
				(gammaPre[std::min(intensityG, gammaPreSize)] * (gmask & 0x01010101)) +
				(gammaPre[std::min(intensityB, gammaPreSize)] * (bmask & 0x01010101)) +
				amask;
		}
}

void DrawFps(int x, int y)
{
	static bool isFirstFrame=true;
	static unsigned frameCnt =0;
	static double displayTime;
	static unsigned updateTime;
	
	if (isFirstFrame) {
		isFirstFrame=false;
		updateTime = ps.curTime;		
		displayTime = 0;
		return;
		}		
	if (!ps.isPaused)
		frameCnt++;
		
	if (ps.curTime-updateTime>300 && frameCnt>0){
		displayTime = (ps.curTime-updateTime)/frameCnt;		
		updateTime = ps.curTime;
		frameCnt=0;
		}
	
	DrawString(x,y,"%5.1f fps  %5.1f ms  ", 1000/displayTime, displayTime);	
	//printf("%5.1f  ",1000/displayTime);
}

int CreateMovingPoints(Camera cam, int processPointsCount)
{				
	if(ps.isPaused ) return 0;
	
	gridSize = 130.0f/pow(1.0f*ps.nx*ps.ny*ps.nz, 1.f/3);
	
	const int gridVectorN = int(gridVector.size());						
	//ps.pointCntExp=-10;
	float pointCntInv=0.11f*(1<<(-ps.pointCntExp));	
	pointCntInv *= sqrtf(sqrtf(float(ps.nx*ps.ny*ps.nz)))/40;
	
	Vec gridCenter = VecMul( Vec(ps.nx/2.0f-0.5f, ps.ny/2.0f-0.5f, ps.nz/2.0f-0.5f), gridSize);
	
	const float cameraDistOuterFact = 4/(cameraArrange.dist+60); 
	
	const unsigned minMsBetweenPoints = 4000/cMaxDotsPerLine;
	
	int createdMpI=0;
		
	if (processPointsCount>gridVectorN) processPointsCount=gridVectorN;		
	static int i=0;
	
	for(int processPointsI=0; processPointsI<processPointsCount; processPointsI++, i+=1) {
		if (i>=gridVectorN) {i=0;}
		VectorData& gridVec = gridVector[i];						
		float offsetFrac= useSpeed ? (VecLen(gridVec.v)/500) : (1.0/4000);
			
		unsigned ib = (gridVec.indices>>42) & 0x1FFFFF;
		unsigned jb = (gridVec.indices>>21) & 0x1FFFFF;
		unsigned kb = gridVec.indices&0x1FFFFF;
		Vec beg = Vec(gridSize*ib - gridCenter.x, gridSize*jb - gridCenter.y, gridSize*kb - gridCenter.z);
		
		Vec cameraToBeg = VecSub(beg,cam.eye);
		float zdist = DotProduct( cameraToBeg,  cam.dir);
		float distSqr = DotProduct(cameraToBeg, cameraToBeg);		
		float densityInvFactor = zdist;			
		if (zdist<2)
			densityInvFactor = distSqr*0.5f;
		else if(zdist*zdist*1.26f<distSqr)
			densityInvFactor = (distSqr + 160)*cameraDistOuterFact;	
		unsigned msBetweenPoints = unsigned(pointCntInv * densityInvFactor * gridVec.dotSpread);
		if (msBetweenPoints < minMsBetweenPoints) 
			msBetweenPoints = minMsBetweenPoints;
				
		unsigned time = ps.curTime - gridVec.lastMpTime;
		time=unsigned(float(time) / gridVec.prevMsBetweenPoints * msBetweenPoints);
		gridVec.prevMsBetweenPoints = msBetweenPoints;
		
		if (time>=msBetweenPoints) {
			MovingPoint mp;							
			mp.v = gridVec.v;
			mp.s = beg;
			
			if (time>=msBetweenPoints+msBetweenPoints+msBetweenPoints) {				
				mp.brightness = gridVec.dotBrightness*(time-(msBetweenPoints+msBetweenPoints));
				time = msBetweenPoints+msBetweenPoints;
				mp.offset = time*offsetFrac;				
				createdMp[createdMpI] = mp;
				createdMpI++;
				}
			for (int k=0;k<2;k++) {
				time -= msBetweenPoints;
				mp.offset = time*offsetFrac;	
				mp.brightness = gridVec.dotBrightness*msBetweenPoints;
				createdMp[createdMpI] = mp;
				createdMpI++;				
				if (time<msBetweenPoints) break;
				}			
			}		
		gridVec.lastMpTime = ps.curTime-time;
		}		

	return createdMpI;
}

void SaveScreenshot(char *filename)
{
	SDL_Rect rect;
	rect.x = rect.y = 0;
	rect.w = screenW;
	rect.h = screenH;

	SDL_Surface* screen24 = SDL_CreateRGBSurface(0, screenW, screenH, 24, rmask, gmask, bmask, 0);
	SDL_BlitSurface(screen, &rect, screen24, &rect);

	if(filename){
		SDL_SaveBMP(screen24, filename);
	}
	else{
		static int cnt=0;
		cnt++;

		time_t rawtime;
		time ( &rawtime );
		struct tm * timeinfo;
		timeinfo = localtime(&rawtime);
		char outFileName[100];
	
		sprintf(outFileName,"vv_capture_%02d%02d%02d-%02d%02d_%03d.bmp",
		timeinfo->tm_year % 100, timeinfo->tm_mon+1, timeinfo->tm_mday,
		timeinfo->tm_hour,timeinfo->tm_min, cnt );    
		SDL_SaveBMP(screen24, outFileName);
	}
	SDL_FreeSurface(screen24);
}

void SeparateMovingPoints(int n)
{	
	for (int i=0;i<n;i++){
		MovingPoint& mp = createdMp[i];
		separatedMp[i+0*n] = mp.s.x;
		separatedMp[i+1*n] = mp.s.y;
		separatedMp[i+2*n] = mp.s.z;
		separatedMp[i+3*n] = mp.v.x;
		separatedMp[i+4*n] = mp.v.y;
		separatedMp[i+5*n] = mp.v.z;		
		separatedMp[i+6*n] = mp.offset;
		separatedMp[i+7*n] = mp.brightness;
		}
}

void DrawSDL()
{					
	Camera cam = CreateCamera(cameraArrange);
		
	TIMER ktm;
	ktm.Reset();	
	
	if (showVectors)
	{
		CallMovingPointsRenderer(
			cam,
			cudaMem.mpData,
			cudaMem.mpBufDesc,
			int( mpBufDescCuda.size() ),
			cudaMem.intensityRaster,
			screenW, screenH,
			ps.curTime,
			ps.exposure/ps.totalBrightness*cBrightnessMultiplier,
			ps.length*cLengthMultiplier,
			maxLength * colorScale,
			useColor,
			useSpeed
			);
	}
	if (showHeatmap)
	{
		CallQuadRenderer(
			cam,
			cudaMem.quadData,
			cudaMem.quadBufDesc,
			(int)quadBufDescCuda.size(),
			cudaMem.intensityRaster,
			screenW,
			screenH,
			ps.exposure/ps.totalBrightness*cBrightnessMultiplier,
			maxLength * colorScale,
			gridSize
			);
	}

	cudaStreamQuery(0);
	
	const int processVectorsCount = 600*1000;
		
	static int clearThrottle=0;
	clearThrottle--;	
	if (clearThrottle<0) clearThrottle=0;
	// reduce the number of points if the GPU moving point buffer is not sufficient
	if ( clearThrottle==0 && mpBufDescCuda.size()*processVectorsCount*3>cudaMaxPoints 
		 && !ps.isPaused && ps.curTime > mpBufDescCuda.back().startTime && 
		 cudaMaxPoints/4000 <= mpBufDescCuda.back().n/(ps.curTime - mpBufDescCuda.back().startTime) 
		 )
			 { ps.pointCntExp--; clearThrottle=(ps.nx*ps.ny)/processVectorsCount+1; }

	int newMpN = CreateMovingPoints(cam, processVectorsCount);
	
	if (clearThrottle>0) 
		newMpN=0;
	
	int createTime=ktm.Get();					
	
	SeparateMovingPoints(newMpN);				

	static int mpDataI=0;
	if (mpDataI + newMpN >= cudaMaxPoints)
		mpDataI = 0;

	SMpBufDesc curMpBufDescCuda;
	curMpBufDescCuda.beg = mpDataI * (sizeof(MovingPoint)/sizeof(float));
	curMpBufDescCuda.n = newMpN;
	curMpBufDescCuda.startTime = ps.curTime;
		
	mpBufDescCuda.push_back(curMpBufDescCuda);
	mpDataI += newMpN;
	
	while (ps.curTime - mpBufDescCuda[0].startTime > 4000)
		mpBufDescCuda.erase(mpBufDescCuda.begin() );			
	
	cudaMemcpy(intRaster.pix, cudaMem.intensityRaster, screenW*screenH*sizeof(unsigned)*3, cudaMemcpyDeviceToHost);
	int kerTime=ktm.Get();	
	
	RenderIntensityRaster();
	
	DrawString(-4,0," %-.32s", inFileName);
	DrawString(-4,24," %d x %d x %d", ps.nx, ps.ny, ps.nz);
		
	DrawString(screenW-585, 0," Density: %3d  o/p   ", ps.pointCntExp);	
	DrawString(screenW-314, 0," Length  : %5.1f     a/d ", ps.length);
	DrawString(screenW-314,24," Exposure: %5.1f     w/s ", ps.exposure);
	DrawString(screenW-314,48," CamDist : %5.1f  Num5/8 ", cameraArrange.dist);
	DrawString(screenW-236,72," Auto Adjust %s m ", ps.isAutoAdjust? "ON " : "OFF");

	if (showParams){
		DrawString(0, 24,"centerX: %2.1f", cameraArrange.centerTranslation.x);
		DrawString(0, 48,"centerY: %2.1f", cameraArrange.centerTranslation.y);
		DrawString(0, 72,"centerZ: %2.1f", cameraArrange.centerTranslation.z);
		DrawString(0, 96,"distance: %2.1f", cameraArrange.dist);
		DrawString(0, 120,"rotLR: %2.1f", cameraArrange.rotLR);
		DrawString(0, 144,"rotUD: %2.1f", cameraArrange.rotUD);
		DrawString(0, 168,"exposure: %2.1f", ps.exposure);
		DrawString(0, 192,"length: %2.1f", ps.length);
		DrawString(0, 216,"intensity: %d", ps.pointCntExp);
		DrawString(0, 240,"colorScale: %2.2f", colorScale);
		DrawString(0, 264,"time: %d", ps.curTime);

		fprintf(stdout, "centerX: %2.1f\n", cameraArrange.centerTranslation.x);
		fprintf(stdout, "centerY: %2.1f\n", cameraArrange.centerTranslation.y);
		fprintf(stdout, "centerZ: %2.1f\n", cameraArrange.centerTranslation.z);
		fprintf(stdout, "distance: %2.1f\n", cameraArrange.dist);
		fprintf(stdout, "rotLR: %2.1f\n", cameraArrange.rotLR);
		fprintf(stdout, "rotUD: %2.1f\n", cameraArrange.rotUD);
		fprintf(stdout, "exposure: %2.1f\n", ps.exposure);
		fprintf(stdout, "length: %2.1f\n", ps.length);
		fprintf(stdout, "intensity: %d\n", ps.pointCntExp);
		fprintf(stdout, "colorScale: %2.2f\n", colorScale);
		fprintf(stdout, "time: %d\n", ps.curTime);
	}

	
	
	if (ps.isScreenshotRequest) {
		void SaveScreenshot(char *filename);
		SaveScreenshot(NULL);
		ps.isScreenshotRequest=false;		
	}
		
	DrawFps(screenW-585,24);
	
		
	SDL_UpdateWindowSurface(display);
	cudaMemcpy(cudaMem.mpData+curMpBufDescCuda.beg, &separatedMp[0], newMpN*sizeof(MovingPoint), cudaMemcpyHostToDevice);		
	cudaMemcpy(cudaMem.mpBufDesc, &mpBufDescCuda[0], mpBufDescCuda.size()*sizeof(SMpBufDesc), cudaMemcpyHostToDevice);					
	ClearIntensityRasterCuda();


	if (dumpFilename){
		if (ps.curTime >= timeToSimulate){
			SaveScreenshot(dumpFilename);
			exit(0);
		}
	}
}

void EventLoop()
{	
	int sym;
	SDL_Event event;
		bool isEvent=false;
	bool isDrawReq=true;
	int noDrawEventCount=0;
	TIMER lastDrawTime;
	TIMER lastSimStepTime;
	TIMER noRedrawTimer;
		
	lastDrawTime.Reset();
	lastSimStepTime.Reset();
	noRedrawTimer.Reset();

	while ( !ps.sys.isQuit) {				
		if (lastDrawTime.Get()>=9)
			isDrawReq = true;
		
		isEvent= SDL_PollEvent(&event) !=0;				
		if (!isEvent && !isDrawReq) {
			SDL_Delay(1);
			noDrawEventCount=990;
			}						
		noDrawEventCount++;		
		if (!(isEvent && noDrawEventCount<1000) && isDrawReq && lastDrawTime.Get()>=20) {
			unsigned timePassed = lastSimStepTime.Get();			
			if (!ps.isPaused && ps.sys.isActive)
				AnimationStep(timePassed);			
			lastSimStepTime.Subtract(timePassed);
			if (noRedrawTimer.Get()>200) {
				lastDrawTime.Reset();
				DrawSDL();		
				isDrawReq=false;
				}
				
			noDrawEventCount=0;			
			}
		else if(!isEvent)
			SDL_Delay(1);
		if (noRedrawTimer.Get()>2*1000*1000)
			noRedrawTimer.Subtract(1000*1000);
		
		{
		float prevDist = cameraArrange.dist;
		if (ps.sys.isMouseLeftPressed){
			if (noRedrawTimer.Get()>200) {
				cameraArrange.dist-=ps.pressMouseLeftTime.Get()*0.0007f*cameraArrange.dist;
				if (cameraArrange.dist<1.0) cameraArrange.dist=1.0;			
				isDrawReq=true;
				}
			ps.pressMouseLeftTime.Reset();			
			}
		if (ps.sys.isMouseRightPressed){
			if (noRedrawTimer.Get()>200) {
				cameraArrange.dist+=ps.pressMouseRightTime.Get()*0.0007f*cameraArrange.dist;				
				isDrawReq=true;
				}
			ps.pressMouseRightTime.Reset();
			}
		
		if (ps.sys.isKeyPadUp){
			cameraArrange.dist -= ps.pressKeyPadUpTime.Get()*0.0001f*cameraArrange.dist;			
			ps.pressKeyPadUpTime.Reset();
			isDrawReq=true;
			}
		if (ps.sys.isKeyPadDown){
			cameraArrange.dist += ps.pressKeyPadDownTime.Get()*0.0001f*cameraArrange.dist;			
			ps.pressKeyPadDownTime.Reset();
			isDrawReq=true;
			}
		if (ps.isAutoAdjust && !dumpFilename){
			float changeFactor = cameraArrange.dist/prevDist;
			ps.length *= sqrtf(changeFactor);
			ps.exposure *= sqrtf(changeFactor);
			}
		}
		{
		Camera cam = CreateCamera(cameraArrange);
		Vec fwdDirection = VecUnit(Vec(cam.dir.x, 0.0f, cam.dir.z));		
		Vec lftDirection = Vec(fwdDirection.z, 0.0f, -fwdDirection.x);
		float speedFactor = cameraArrange.dist*0.0002f;
		
		if (ps.sys.isKeyRightPressed){
			Vec trans = VecMul(lftDirection, ps.pressKeyRightTime.Get()*speedFactor);
			cameraArrange.centerTranslation = VecAdd(cameraArrange.centerTranslation, trans);
			ps.pressKeyRightTime.Reset();			
			isDrawReq=true;
			}
		if (ps.sys.isKeyLeftPressed){
			Vec trans = VecMul(lftDirection, ps.pressKeyLeftTime.Get()*-speedFactor);
			cameraArrange.centerTranslation = VecAdd(cameraArrange.centerTranslation, trans);			
			ps.pressKeyLeftTime.Reset();			
			isDrawReq=true;
			}
		if (ps.sys.isKeyUpPressed){			
			Vec trans = VecMul(fwdDirection, ps.pressKeyUpTime.Get()*speedFactor);
			cameraArrange.centerTranslation = VecAdd(cameraArrange.centerTranslation, trans);			
			ps.pressKeyUpTime.Reset();			
			isDrawReq=true;
			}
		if (ps.sys.isKeyDownPressed){						
			Vec trans = VecMul(fwdDirection, ps.pressKeyDownTime.Get()*-speedFactor);
			cameraArrange.centerTranslation = VecAdd(cameraArrange.centerTranslation, trans);
			ps.pressKeyDownTime.Reset();			
			isDrawReq=true;
			}
		}
		if (ps.sys.isKeyPgDnPressed){												
			ps.exposure-=ps.pressKeyPgDnTime.Get()*cBrightnessStep*ps.exposure;
			ps.pressKeyPgDnTime.Reset();
			if (ps.exposure<cBrightnessMin) ps.exposure = cBrightnessMin;
			isDrawReq=true;
			}
		if (ps.sys.isKeyPgUpPressed){												
			ps.exposure+=ps.pressKeyPgUpTime.Get()*cBrightnessStep*ps.exposure;
			ps.pressKeyPgUpTime.Reset();
			if (ps.exposure>cBrightnessMax) ps.exposure = cBrightnessMax;
			isDrawReq=true;
			}
		if (ps.sys.isKeyHomePressed){												
			ps.length+=ps.pressKeyHomeTime.Get()*cLengthStep*ps.length;
			ps.pressKeyHomeTime.Reset();
			if (ps.length>cLengthMax) ps.length = cLengthMax;
			isDrawReq=true;
			}
		if (ps.sys.isKeyEndPressed){												
			ps.length-=ps.pressKeyEndTime.Get()*cLengthStep*ps.length;
			ps.pressKeyEndTime.Reset();
			if (ps.length<cLengthMin) ps.length = cLengthMin;
			isDrawReq=true;
			}
		if (ps.sys.isKeyPadLeft){
			cameraArrange.rotLR -= ps.pressKeyPadLeftTime.Get() *0.01f;
			ps.pressKeyPadLeftTime.Reset();
			isDrawReq=true;
			}
		if (ps.sys.isKeyPadRight){
			cameraArrange.rotLR += ps.pressKeyPadRightTime.Get() *0.01f;
			ps.pressKeyPadRightTime.Reset();
			isDrawReq=true;
			}
		
		if (!isEvent) 
			continue;
		switch(event.type){
			case SDL_WINDOWEVENT:
				switch(event.window.event){
					case SDL_WINDOWEVENT_RESIZED:
						SDL_GetWindowSize(display, &screenW, &screenH);
						screen = SDL_GetWindowSurface(display);
						break;
					case SDL_WINDOWEVENT_EXPOSED:
					case SDL_WINDOWEVENT_SHOWN:
						isDrawReq=true;
						break;
					case SDL_WINDOWEVENT_RESTORED:
						if (!ps.sys.isActive){
							SDL_SetWindowGrab(display, SDL_TRUE);
							SDL_SetRelativeMouseMode(SDL_TRUE);
							SDL_ShowCursor(0);
							noRedrawTimer.Reset();
							ps.sys.isActive=true;
						}
						break;
					case SDL_WINDOWEVENT_MINIMIZED:
						if (ps.sys.isActive){
							SDL_SetWindowGrab(display, SDL_FALSE);
							SDL_SetRelativeMouseMode(SDL_FALSE);
							SDL_ShowCursor(1);
							ps.sys.isActive=false;
						}

						break;
				}
				break;
			case SDL_QUIT: 
				ps.sys.isQuit = true;
				break;							
			case SDL_MOUSEMOTION:
				if (dumpFilename) break;
				if(!ps.sys.isActive) break;
				if (noRedrawTimer.Get()>50) {
					cameraArrange.rotLR+=event.motion.xrel*0.05f;
					if (cameraArrange.rotLR>=360) cameraArrange.rotLR-=360;
					if (cameraArrange.rotLR <0)   cameraArrange.rotLR+=360;
					cameraArrange.rotUD+=event.motion.yrel*0.05f;
					if (cameraArrange.rotUD> 89.9f) cameraArrange.rotUD= 89.9f;
					if (cameraArrange.rotUD<-89.9f) cameraArrange.rotUD=-89.9f;				
					isDrawReq=true;				
					}
				break;							
			case SDL_MOUSEBUTTONDOWN:			    
				if (event.button.button==SDL_BUTTON_LEFT){
					ps.sys.isMouseLeftPressed=true;
					ps.pressMouseLeftTime.Reset();
					}
				if (event.button.button==SDL_BUTTON_RIGHT){
					ps.sys.isMouseRightPressed=true;
					ps.pressMouseRightTime.Reset();
					}				
				break;							
			case SDL_MOUSEBUTTONUP:
				if (event.button.button==SDL_BUTTON_LEFT){
					ps.sys.isMouseLeftPressed=false;					
					}
				if (event.button.button==SDL_BUTTON_RIGHT){
					ps.sys.isMouseRightPressed=false;					
					}				
				break;
			case SDL_KEYDOWN:
				sym=event.key.keysym.sym;										
				if ( sym== SDLK_ESCAPE ) {
					ps.sys.isQuit = true;
					break;
					}
				if (sym==SDLK_KP_0){
					cameraArrange.centerTranslation = Vec(0,0,0);
					}
				if (sym==SDLK_LEFT){
					ps.sys.isKeyLeftPressed=true;
					ps.pressKeyLeftTime.Reset();										
					}  
				if (sym==SDLK_UP)  {
					ps.sys.isKeyUpPressed=true;
					ps.pressKeyUpTime.Reset();										
					}    
				if (sym==SDLK_DOWN) {
					ps.sys.isKeyDownPressed=true;
					ps.pressKeyDownTime.Reset();										
					}  
				if (sym==SDLK_RIGHT){
					ps.sys.isKeyRightPressed=true;
					ps.pressKeyRightTime.Reset();										
					}
				if (sym==SDLK_SPACE){
					ps.isPaused=!ps.isPaused;					
					}
				if (sym==SDLK_w){
					ps.sys.isKeyPgUpPressed=true;
					ps.pressKeyPgUpTime.Reset();
					}
				if (sym==SDLK_s){
					ps.sys.isKeyPgDnPressed=true;
					ps.pressKeyPgDnTime.Reset();
					}
				if (sym==SDLK_d){
					ps.sys.isKeyHomePressed=true;
					ps.pressKeyHomeTime.Reset();
					}
				if (sym==SDLK_a){
					ps.sys.isKeyEndPressed=true;
					ps.pressKeyEndTime.Reset();
					}
				if (sym==SDLK_t){
					colorScale += 0.01;
					}
				if (sym==SDLK_g){
					if (colorScale > 0.02) colorScale -= 0.01;
					}
				if (sym==SDLK_e){
					cameraArrange.rotLR += 45;
					if (cameraArrange.rotLR >= 360) cameraArrange.rotLR -= 360;
					}
				if (sym==SDLK_q){
					cameraArrange.rotLR -= 45;
					if (cameraArrange.rotLR < 0) cameraArrange.rotLR += 360;
					}
				if (sym==SDLK_u){
					if (displayMode == 0) {
						showVectors = false;
						showHeatmap = true;
					}
					if (displayMode == 1) {
						showHeatmap = false;
						showVectors = true;
					}
					if (displayMode == 2) {
						showHeatmap = true;
						showVectors = true;
					}
					displayMode = (++displayMode % 3);
				}
				if (sym==SDLK_KP_6){
					ps.sys.isKeyPadRight=true;
					ps.pressKeyPadRightTime.Reset();
					}  
				if (sym==SDLK_KP_4)  {
					ps.sys.isKeyPadLeft=true;
					ps.pressKeyPadLeftTime.Reset();
					}    
				if (sym==SDLK_KP_5) {
					ps.sys.isKeyPadDown=true;
					ps.pressKeyPadDownTime.Reset();
					}  
				if (sym==SDLK_KP_8){
					ps.sys.isKeyPadUp=true;
					ps.pressKeyPadUpTime.Reset();
					}
				if (sym==SDLK_p){
					ps.pointCntExp++;					
					if (ps.pointCntExp>0) ps.pointCntExp=0;
					}
				if (sym==SDLK_o){
					ps.pointCntExp--;
					if (ps.pointCntExp<-10) ps.pointCntExp=-10;
					}
				if (sym==SDLK_m){
					ps.isAutoAdjust = !ps.isAutoAdjust;
					}
				break;
			case SDL_KEYUP:
				sym=event.key.keysym.sym;										
				if (sym==SDLK_LEFT){
					ps.sys.isKeyLeftPressed=false;					
					}  
				if (sym==SDLK_UP)  {
					ps.sys.isKeyUpPressed=false;					
					}    
				if (sym==SDLK_DOWN) {
					ps.sys.isKeyDownPressed=false;				
					}  
				if (sym==SDLK_RIGHT){
					ps.sys.isKeyRightPressed=false;					
					}
				if (sym==SDLK_w) {
					ps.sys.isKeyPgUpPressed=false;				
					}  
				if (sym==SDLK_s) {
					ps.sys.isKeyPgDnPressed=false;				
					}
				if (sym==SDLK_d) {
					ps.sys.isKeyHomePressed=false;				
					}
				if (sym==SDLK_a) {
					ps.sys.isKeyEndPressed=false;
					}
				if (sym==SDLK_KP_6){
					ps.sys.isKeyPadRight=false;
					}  
				if (sym==SDLK_KP_4)  {
					ps.sys.isKeyPadLeft=false;
					}
				if (sym==SDLK_KP_5) {
					ps.sys.isKeyPadDown=false;
					}
				if (sym==SDLK_KP_8){
					ps.sys.isKeyPadUp=false;			
					}
				if (sym==SDLK_PRINTSCREEN){
					ps.isScreenshotRequest=true;
					}					
				break;				
			}
		}
	
}

void Init() 
{					
	createdMp.resize(gridVector.size()*3);
	separatedMp.resize(gridVector.size()*3*sizeof(MovingPoint)/sizeof(float));
	
	cameraArrange.rotLR = rotLRInit;
	cameraArrange.rotUD = rotUDInit;
	cameraArrange.dist = distInit;
	cameraArrange.centerTranslation = Vec(centerXInit, centerYInit, centerZInit);
	
	ps.exposure = brightnessInit;
	ps.length = lengthInit;	
	ps.pointCntExp = initialDotDensity;
	ps.isAutoAdjust=true;
		
	//calculate maximum number of miliseconds between dots of a vector
	double sumLen=0;
	for (int i=0;i<gridVector.size();i++)
		sumLen += VecLen(gridVector[i].v);			
	float maxDotSpread = 1.0f/float(0.01*sumLen/(gridVector.size()));
	
	for (int i=0;i<gridVector.size();i++)
		gridVector[i] = CreateGridVector(gridVector[i].v, gridVector[i].indices, maxDotSpread);
	
	//normalize total brightness
	double sumBrightness = 0;
	for (int i=0;i<gridVector.size();i++)
		sumBrightness += gridVector[i].dotBrightness;
	ps.totalBrightness = float(sumBrightness)/(screenW*screenW);
	
	if (!dumpFilename){
		//reduce initial exposure on small grids
		if ( sqrt(1.0*ps.nx*ps.ny*ps.nz)<700 )
			ps.exposure *= float(sqrt(1.0*ps.nx*ps.ny*ps.nz)/700);
	}
		
	ps.sys.isActive=true;
	ps.sys.isQuit=false;
	ps.sys.isMouseLeftPressed=false;
	ps.sys.isMouseRightPressed=false;
	ps.sys.isKeyLeftPressed=false;
	ps.sys.isKeyRightPressed=false;
	ps.sys.isKeyUpPressed=false;
	ps.sys.isKeyDownPressed=false;
	ps.sys.isKeyPgDnPressed=false;
	ps.sys.isKeyPgUpPressed=false;
	ps.sys.isKeyHomePressed=false;
	ps.sys.isKeyEndPressed=false;
	ps.sys.isKeyPadUp=false;
	ps.sys.isKeyPadDown=false;
	ps.sys.isKeyPadLeft=false;
	ps.sys.isKeyPadRight=false;
	
	ps.isPaused=false;	
	ps.isScreenshotRequest=false;
	ps.curTime = 0;
	ps.programTime.Reset();
}

int InitCuda()
{
	int idev;
	cudaDeviceProp deviceProp;
	cudaError_t cudaError;

	idev = 0;
	cudaError=cudaSetDevice(idev);
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Could not find nvidia GPU device 0\n");
		return -1;
		}
	cudaError=cudaGetDeviceProperties(&deviceProp, idev);		
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Nvidia GPU device 0 not responding\n");
		return -1;
		}	
	
	cudaError=cudaMalloc((void**)&cudaMem.intensityRaster, screenW*screenH*sizeof(unsigned)*3); 
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Out of GPU device memory\n");
		return -1;
		}	
	cudaError=cudaMalloc((void**)&cudaMem.mpData, cudaMaxPoints*sizeof(MovingPoint)); 
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Out of GPU device memory\n");
		return -1;
		}
	cudaError=cudaMalloc((void**)&cudaMem.mpBufDesc, 2000*sizeof(SMpBufDesc)); 
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Out of GPU device memory\n");
		return -1;
		}	
	cudaError=cudaMalloc((void**)&cudaMem.quadBufDesc, quadBufDescCuda.size() * sizeof(SQuadBufDesc));
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Out of GPU device memory\n");
		return -1;
		}
	cudaMemcpy(cudaMem.quadBufDesc, quadBufDescCuda.data(), quadBufDescCuda.size() * sizeof(SQuadBufDesc), cudaMemcpyHostToDevice);
	cudaError=cudaMalloc((void**)&cudaMem.quadData, quadData.size() * sizeof(float));
	if (cudaError!=cudaSuccess){
		fprintf(stderr,"Out of GPU device memory\n");
		return -1;
		}
	cudaMemcpy(cudaMem.quadData, quadData.data(), quadData.size() * sizeof(float), cudaMemcpyHostToDevice);
	
	return 0;
}

char *GetCmdOption(char **begin, char **end, const std::string &option)
{
	char **iFind = std::find(begin, end, option);
	if (iFind != end && ++iFind != end){
		return *iFind;
	}
	return 0;
}

bool CmdOptionExists(char **begin, char **end, const std::string &option)
{
	return std::find(begin, end, option) != end;
}

void OverrideOption(char *&option, char *cmdOption)
{
	if (cmdOption) option = cmdOption;
}

void indent(size_t amount)
{
	for (size_t i = 0; i != amount; ++i)
	{
		cout << '\t';
	}
}

void showYAMLValue(Sip::YAMLDocumentUTF8::Node *node, size_t amount = 0)
{
	while (node)
	{
		indent(amount);
		cout << "Node: ";
		switch (node->type() & 0xF)
		{
		case Sip::Sequence:
			cout << "Sequence";
			break;
		case Sip::Mapping:
			cout << "Mapping";
			break;
		case Sip::Comment:
			cout << "Comment";
		}
		cout << std::endl;
		if (node->key())
		{
			indent(amount);
			cout << "  Key: \"";
			cout.write(node->key(), node->keySize());
			cout << "\"" << std::endl;
		}
		if (node->value())
		{
			indent(amount);
			cout << "  Value: \"";
			cout.write(node->value(), node->valueSize());
			cout << "\"" << std::endl;
		}
		if (node->firstChild())
		{
			showYAMLValue(node->firstChild(), amount + 1);
		}
		node = node->nextSibling();
	}
}

int main(int argc, char **argv) 
{  		
	if (argc<2){
		#ifdef USE_DEFAULT_FILE
			strcpy(inFileName,inFileNameDefault);
		#else
			printf("%s\n", programName);
			printf("Usage:\n");
			printf("\tvector_viz  inputFile\n");
			printf("\tvector_viz  inputFile.vvt  -c  convertedFile.vvf\n");			
			printf("\tvector_viz  inputFile -params\n");
			printf("\tvector_viz  inputFile -centerX <val> -centerY <val> -centerZ <val> -rotLR <val> -rotUD <val> -distance <val> -exposure <val> -length <val> -intensity <val> -time <val> -dump <filename.bmp>\n");
			printf("\tvector_viz  inputFile -offscreen -w <val> -h <val>\n");
			printf("\tvector_viz  inputFile -color -speed\n");
			printf("\tvector_viz  -yaml input.yaml\n");
			return 0;
		#endif		
		}
	else
		strcpy(inFileName,argv[1]);
		
	char *centerX = nullptr;
	char *centerY = nullptr;
	char *centerZ = nullptr;
	char *rotLR = nullptr;
	char *rotUD = nullptr;
	char *distance = nullptr;
	char *exposure = nullptr;
	char *length = nullptr;
	char *intensity = nullptr;
	char *colorscale = nullptr;
	char *time = nullptr;
	char *dump = nullptr;
	char *inPlane = nullptr;
	char *w = nullptr;
	char *h = nullptr;

	if (CmdOptionExists(argv, argv + argc, "-yaml")) {
		char *yamlFilename = GetCmdOption(argv, argv + argc, "-yaml");
		strcpy(inFileName,yamlFilename);
		char *yamlSource = NULL;
		FILE *yamlFile = fopen(yamlFilename,"r");
		if (yamlFile == NULL){
			fprintf(stdout,"Error opening yaml file: %s\n", yamlFilename);
			return 1;
		}
		fseek(yamlFile, 0L, SEEK_END);
		long size = ftell(yamlFile);
		fseek(yamlFile, 0, SEEK_SET);
		yamlSource = (char*) malloc(size + 1);
		size = fread(yamlSource, 1, size, yamlFile);
		yamlSource[size] = 0;
		fclose(yamlFile);

		Sip::YAMLDocumentUTF8 doc;
		doc.parse(yamlSource);
		//showYAMLValue(doc.firstChild());
		Sip::YAMLDocumentUTF8::Node *pInputNode;
		Sip::YAMLDocumentUTF8::Node *pOutputNode;

		for (auto pNode = doc.firstChild(); pNode; pNode = pNode->nextSibling())
		{
			if (pNode->type() != Sip::YAMLType::Mapping) continue;

			((char*)pNode->key())[pNode->keySize()] = 0;
			if (!strcmp("input", pNode->key()))
			{
				pInputNode = pNode;
			}
			else if (!strcmp("output", pNode->key()))
			{
				pOutputNode = pNode;
			}
		}

		char *nameRoot = nullptr;
		char *dir = nullptr;
		char *ngxStr = nullptr;
		char *ngyStr = nullptr;
		char *ngzStr = nullptr;
		char *snxStr = nullptr;
		char *snyStr = nullptr;
		char *snzStr = nullptr;
		char *xcutStr = nullptr;
		char *ycutStr = nullptr;
		char *zcutStr = nullptr;

		std::pair<char const*, char**> inParameters[] =
		{
			std::make_pair("name_root", &nameRoot),
			std::make_pair("dir",       &dir),
			std::make_pair("ngx",       &ngxStr),
			std::make_pair("ngy",       &ngyStr),
			std::make_pair("ngz",       &ngzStr),
			std::make_pair("snx",       &snxStr),
			std::make_pair("sny",       &snyStr),
			std::make_pair("snz",       &snzStr),
			std::make_pair("xcut",      &xcutStr),
			std::make_pair("zcut",      &zcutStr),
			std::make_pair("ycut",      &ycutStr),
		};

		for (auto pNode = pInputNode->firstChild(); pNode; pNode = pNode->nextSibling())
		{
			if (pNode->type() != Sip::YAMLType::Mapping) continue;

			((char*)pNode->key())[pNode->keySize()] = 0;
			for (auto &parameter: inParameters)
			{
				if (!strcmp(parameter.first, pNode->key()))
				{
					if (pNode->valueSize())
					{
						*parameter.second = (char*)pNode->value();
						(*parameter.second)[pNode->valueSize()] = 0;
					}
				}
			}
		}

		if (!nameRoot)
		{
			fprintf(stdout, "Please specify input name root\n");
			return 1;
		}

		if (!dir)
		{
			fprintf(stdout, "Please specify input directory\n");
			return 1;
		}

		auto ngx = atoi(ngxStr);
		auto ngy = atoi(ngyStr);
		auto ngz = atoi(ngzStr);
		auto snx = atoi(snxStr);
		auto sny = atoi(snyStr);
		auto snz = atoi(snzStr);
		Sip::YAMLDocumentUTF8::Node *pOrthoNode = nullptr;
		Sip::YAMLDocumentUTF8::Node *pAngleNode = nullptr;

		char *outNameRoot = nullptr;
		char *outFilename = nullptr;
		std::pair<char const*, char**> outParameters[] =
		{
			std::make_pair("w",         &w),
			std::make_pair("h",         &h),
			std::make_pair("centerX",   &centerX),
			std::make_pair("centerY",   &centerY),
			std::make_pair("centerZ",   &centerZ),
			std::make_pair("rotLR",     &rotLR),
			std::make_pair("rotUD",     &rotUD),
			std::make_pair("distance",  &distance),
			std::make_pair("exposure",  &exposure),
			std::make_pair("length",    &length),
			std::make_pair("intensity", &intensity),
			std::make_pair("colorScale",&colorscale),
			std::make_pair("time",      &time),
			std::make_pair("dump",      &dump),
			std::make_pair("name_root", &outNameRoot),
			std::make_pair("filename",  &outFilename),
		};
		std::pair<char const*, bool*> flags[] =
		{
			std::make_pair("params", &showParams),
			std::make_pair("color", &useColor),
			std::make_pair("offscreen", &offScreen),
			std::make_pair("speed", &useSpeed),
			std::make_pair("inPlane", &useInPlane),
			std::make_pair("heatmap", &showHeatmap),
			std::make_pair("vectors", &showVectors),
			std::make_pair("orthographic", &useOrtho),
		};
		std::pair<char const *, Sip::YAMLDocumentUTF8::Node**> subNodes[] =
		{
			std::make_pair("ortho", &pOrthoNode),
			std::make_pair("angle", &pAngleNode),
		};

		for (auto pNode = pOutputNode->firstChild(); pNode; pNode = pNode->nextSibling())
		{
			if (pNode->type() != Sip::YAMLType::Mapping) continue;

			((char*)pNode->key())[pNode->keySize()] = 0;
			for (auto &parameter: outParameters)
			{
				if (!strcmp(parameter.first, pNode->key()))
				{
					if (pNode->valueSize())
					{
						*parameter.second = (char*)pNode->value();
						(*parameter.second)[pNode->valueSize()] = 0;
					}

					break;
				}
			}

			for (auto &flag: flags)
			{
				if (!strcmp(flag.first, pNode->key()))
				{
					((char*)pNode->value())[pNode->valueSize()] = 0;
					*flag.second = !strcmp("true", pNode->value());
					break;
				}
			}

			for (auto &node: subNodes)
			{
				if (!strcmp(node.first, pNode->key()))
				{
					((char*)pNode->value())[pNode->valueSize()] = 0;
					*node.second = pNode;
					break;
				}
			}
		}

		showHeatmap ^= CmdOptionExists(argv, argv + argc, "-heatmap");
		showVectors ^= CmdOptionExists(argv, argv + argc, "-vectors");

		int plane = 0;
		std::vector<int> position;
		int axis = 0;
		std::vector<int> phi;
		bool angle = false;

		struct
		{
			Sip::YAMLDocumentUTF8::Node *pNode;
			char const *pAxisName;
			char const *pPositionName;
			int *pAxis;
			std::vector<int> *pPosition;
		} orientations[] =
		{
			{ pOrthoNode, "plane", "position", &plane, &position },
			{ pAngleNode, "axis",  "phi",      &axis,  &phi },
		};

		for (auto &orientation: orientations)
		{
			if (!orientation.pNode) continue;

			for (auto pNode = orientation.pNode->firstChild(); pNode; pNode = pNode->nextSibling())
			{
				if (pNode->type() != Sip::YAMLType::Mapping) continue;

				((char*)pNode->key())[pNode->keySize()] = 0;
				if (!strcmp(orientation.pAxisName, pNode->key()))
				{
					*orientation.pAxis = pNode->value()[0] - 'x';
				}
				else if (!strcmp(orientation.pPositionName, pNode->key()))
				{
					if (pNode->valueSize())
					{
						((char*)pNode->value())[pNode->valueSize()] = 0;
						orientation.pPosition->push_back(atoi(pNode->value()));
					}
					else
					{
						for (auto pElement = pNode->firstChild(); pElement; pElement = pElement->nextSibling())
						{
							if (pElement->type() != Sip::YAMLType::Sequence) continue;

							if (pElement->valueSize())
							{
								((char*)pElement->value())[pElement->valueSize()] = 0;
								orientation.pPosition->push_back(atoi(pElement->value()));
							}
						}
					}
				}
			}
		}

		bool isOk = ReadMergeInput(dir, nameRoot, ngx, ngy, ngz, snx, sny, snz, plane, position, axis, phi, outFilename, outNameRoot);
		if (!isOk)
		{
			fprintf(stdout,"Error opening input file(s)\n");
			return 1;
		}
	}
	else {
	showHeatmap ^= CmdOptionExists(argv, argv + argc, "-heatmap");
	showVectors ^= CmdOptionExists(argv, argv + argc, "-vectors");
	bool isOk=ReadInputFile(inFileName);
	if (!isOk) {
		fprintf(stdout,"Error opening input file: %s\n", inFileName);		
		return 1;
		}	

		if(CmdOptionExists(argv, argv + argc, "-c")){
			char *outFilename = GetCmdOption(argv, argv + argc, "-c");
			if(outFilename){
				SaveVvfFile(outFilename, gridVector, ps.nz, ps.ny, 1);
				return 0;
			}
			else{
				fprintf(stdout,"Please specify output filename\n");
				return 1;
			}
		}

		auto pCut = strstr(inFileName, "_cut_");
		auto pDeg = strstr(inFileName, "_deg_");
		auto isCut = !!pCut;
		if (!isCut) pCut = pDeg;
		if (pCut)
		{
			auto plane = pCut[5];
			if (pCut[5] && pCut[6] && pCut[7])
			{
				//auto pos = atoi(pCut + 7);
				if (plane == 'y' && isCut)
				{
					for (auto &gridVec: gridVector)
					{
						unsigned y = (gridVec.indices >> 42) & 0x1FFFFF;
						unsigned x = (gridVec.indices >> 21) & 0x1FFFFF;
						unsigned z =  gridVec.indices & 0x1FFFFF;
						gridVec.indices = ((uint64_t)x << 42) + ((uint64_t)y << 21) + z;
					}

					int ny = ps.nx;
					int nx = ps.ny;
					int nz = ps.nz;
					ps.nx = nx;
					ps.ny = ny;
					ps.nz = nz;
				}
				else if ((plane == 'z' && isCut) || (plane == 'x' && !isCut))
				{
					for (auto &gridVec: gridVector)
					{
						unsigned z = (gridVec.indices >> 40) & 0x1FFFFF;
						unsigned x = (gridVec.indices >> 20) & 0x1FFFFF;
						unsigned y =  gridVec.indices & 0x1FFFFF;
						gridVec.indices = ((uint64_t)x << 42) + ((uint64_t)y << 21) + z;
					}

					int nz = ps.nx;
					int nx = ps.ny;
					int ny = ps.nz;
					ps.nx = nx;
					ps.ny = ny;
					ps.nz = nz;
				}
				else if ((plane == 'y' || plane == 'z') && !isCut)
				{
					for (auto &gridVec: gridVector)
					{
						unsigned z = (gridVec.indices >> 40) & 0x1FFFFF;
						unsigned y = (gridVec.indices >> 20) & 0x1FFFFF;
						unsigned x =  gridVec.indices & 0x1FFFFF;
						gridVec.indices = ((uint64_t)x << 42) + ((uint64_t)y << 21) + z;
					}

					int nz = ps.nx;
					int ny = ps.ny;
					int nx = ps.nz;
					ps.nx = nx;
					ps.ny = ny;
					ps.nz = nz;
				}
			}
		}
	}

	OverrideOption(centerX,      GetCmdOption(argv, argv + argc, "-centerX"));
	OverrideOption(centerY,      GetCmdOption(argv, argv + argc, "-centerY"));
	OverrideOption(centerZ,      GetCmdOption(argv, argv + argc, "-centerZ"));
	OverrideOption(rotLR,        GetCmdOption(argv, argv + argc, "-rotLR"));
	OverrideOption(rotUD,        GetCmdOption(argv, argv + argc, "-rotUD"));
	OverrideOption(distance,     GetCmdOption(argv, argv + argc, "-distance"));
	OverrideOption(exposure,     GetCmdOption(argv, argv + argc, "-exposure"));
	OverrideOption(length,       GetCmdOption(argv, argv + argc, "-length"));
	OverrideOption(intensity,    GetCmdOption(argv, argv + argc, "-intensity"));
	OverrideOption(colorscale,   GetCmdOption(argv, argv + argc, "-colorScale"));
	OverrideOption(time,         GetCmdOption(argv, argv + argc, "-time"));
	OverrideOption(dump,         GetCmdOption(argv, argv + argc, "-dump"));

	showParams ^= CmdOptionExists(argv, argv + argc, "-params");
	offScreen  ^= CmdOptionExists(argv, argv + argc, "-offscreen");
	useColor ^= CmdOptionExists(argv, argv + argc, "-color");
	useSpeed ^= CmdOptionExists(argv, argv + argc, "-speed");
	useInPlane ^= CmdOptionExists(argv, argv + argc, "-inPlane");
	useOrtho ^= CmdOptionExists(argv, argv + argc, "-orthographic");

	SetOrtho(useOrtho);

	if ((w || CmdOptionExists(argv, argv + argc, "-w")) && (h || CmdOptionExists(argv, argv + argc, "-h"))){
		OverrideOption(w, GetCmdOption(argv, argv + argc, "-w"));
		OverrideOption(h, GetCmdOption(argv, argv + argc, "-h"));
		screenW = atoi(w);
		screenH = atoi(h);
	}

	if (centerX){
		centerXInit = atof(centerX);
	}

	if (centerY){
		centerYInit = atof(centerY);
	}

	if (centerZ){
		centerZInit = atof(centerZ);
	}

	if (rotLR){
		rotLRInit = atof(rotLR);
	}

	if (rotUD){
		rotUDInit = atof(rotUD);
	}

	if (distance){
		distInit = atof(distance);
	}

	if (exposure){
		brightnessInit = atof(exposure);
	}

	if (length){
		lengthInit = atof(length);
	}

	if (intensity){
		initialDotDensity = atof(intensity);
	}

	if (colorscale){
		colorScale = atof(colorscale);
	}

	if (time){
		timeToSimulate = atoi(time);
	}

	if (dump){
		dumpFilename = strdup(dump);
	}

	//SDL_putenv("SDL_VIDEODRIVER=directx");
	//SDL_putenv("SDL_VIDEODRIVER=dga");
	
	if ( SDL_Init(offScreen ? 0 : SDL_INIT_VIDEO) < 0 ) {
		fprintf(stderr, "Unable to initialize SDL: %s\n", SDL_GetError());
		return 1;
		}

	if (!offScreen){
#if ( /*!defined(_DEBUG) && defined(WIN32)) || */ defined(FULLSCREEN) )
		SDL_DisplayMode current;
		SDL_GetCurrentDisplayMode(0, &current);
		screenW = current.w;
		screenH = current.h;
		display = SDL_CreateWindow(programName, 100, 100, screenW, screenH, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_INPUT_GRABBED);
#else
		display = SDL_CreateWindow(programName, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screenW, screenH, SDL_WINDOW_INPUT_GRABBED);
#endif
		SDL_SetRelativeMouseMode(SDL_TRUE);
	}

	if ( !offScreen && display == NULL ) {
		fprintf(stderr, "Unable to set video mode: %s\n", SDL_GetError());
		SDL_Quit();
		return 2;
	}

	if (offScreen){
		screen   = SDL_CreateRGBSurface(0, screenW, screenH, 32, rmask, gmask, bmask, 0);
	}
	else{
		SDL_GetWindowSize(display, &screenW, &screenH);
		screen = SDL_GetWindowSurface(display);
	}
		
	intRaster.pix=(unsigned*)malloc(screenW*screenH*sizeof(unsigned)*3);
	intRaster.sizeX=screenW;
	intRaster.sizeY=screenH;
	if (intRaster.pix==NULL){
		fprintf(stderr, "Not enough memory for intensity raster.\n");
		SDL_Quit();
		return 3;
		}
	
	SDL_ShowCursor(0);
	
	SetupGammaPre();
	
	UnrollRng();											
	PermutateGridVectors();		
	
	Init();
	
	int retCode = InitCuda();
	if (retCode!=0){
		fprintf(stderr, "Error while initializing cuda: %s\n", "" );
		SDL_Quit();
		return 2;
		}	
	ClearIntensityRasterCuda();
	
	// Loop, drawing and checking events 
	EventLoop();      

	SDL_SetWindowGrab(display, SDL_FALSE);
	SDL_SetRelativeMouseMode(SDL_FALSE);
	SDL_ShowCursor(1);
	
	SDL_Quit();
	return 0;
}  
