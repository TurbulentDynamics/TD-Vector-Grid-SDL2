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

#include "mp_renderer.h"

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

float rotLRInit = 0;
float rotUDInit = -20;
float distInit = 150;
float centerXInit = 0;
float centerYInit = 0;
float centerZInit = 0;
float brightnessInit = 100.0f;
float lengthInit = 100.0f;
float maxLength = 0;
int initialDotDensity = -2;
int timeToSimulate = 0;

char *dumpFilename = NULL;
bool showParams = false;
bool offScreen = false;
bool useColor = false;
bool useSpeed = false;

const char* programName = "VectorViz v1.00";

#ifdef WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>	
	#define USE_DEFAULT_FILE
#endif

#ifdef USE_DEFAULT_FILE
	const char* inFileNameDefault="flowyz_nx_01536_0012000_vect.vvf";
	//const char* inFileNameDefault="flowyz0005030.vvf";
#endif

#if !defined(_DEBUG) && !defined(NO_FULLSCREEN)
	#define FULLSCREEN
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
std::vector<SMpBufDesc> mpBufDesc;
std::vector<SMpBufDesc> mpBufDescCuda;


struct SCudaMemory
{	
	float* mpData;
	SMpBufDesc* mpBufDesc;
	unsigned* intensityRaster;
	
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

struct PointProjection
{
	float x;
	float y;
	float zdistRec;
};

PointProjection PerspProj(Vec t, Camera k)
{	
	PointProjection ret;
	Vec diff=VecSub(t,k.eye);        
	float zdist = DotProduct(diff, k.dir);	
	
	if (zdist < 0.1f) {
		ret.zdistRec = -1;
		return ret;		
		}
	ret.zdistRec=1.0f/zdist;
	Vec proj=VecMul(diff, k.screenDist * ret.zdistRec);		
	proj =VecAdd(proj, k.upLeftCornerTrans);
	ret.x = DotProduct(proj, k.xd);
	ret.y = DotProduct(proj, k.yd);	
	return ret;        
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
	unsigned indices;
	unsigned prevMsBetweenPoints;
	unsigned lastMpTime;
	float dotSpread;		
	float dotBrightness;		
	Vec v;
	//float len;		
};

std::vector<VectorData> gridVector;

VectorData CreateGridVector(Vec inVec, unsigned indices, float maxDotSpread)
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
	for(int i=0; i<ps.nx*ps.ny; i++) {
		VectorData temp = gridVector[i];
		int rndI = unsigned(Rng01()*(ps.nx*ps.ny-i))+i;
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
			gridVector[i].indices= (yi<<16)+xi;
			maxLength = std::max(maxLength, VecLen(gridVector[i].v));
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
	
	const float gridSize = 130.0f/sqrtf(1.0f*ps.nx*ps.ny);												
	
	const int gridVectorN = int(gridVector.size());						
	//ps.pointCntExp=-10;
	float pointCntInv=0.11f*(1<<(-ps.pointCntExp));	
	pointCntInv *= sqrtf(sqrtf(float(ps.ny*ps.ny)))/40;
	
	Vec gridCenter = VecMul( Vec(ps.nx/2.0f-0.5f, 0.0f, ps.ny/2.0f-0.5f), gridSize);			
	
	const float cameraDistOuterFact = 4/(cameraArrange.dist+60); 
	
	const unsigned minMsBetweenPoints = 4000/cMaxDotsPerLine;
	
	int createdMpI=0;
		
	if (processPointsCount>gridVectorN) processPointsCount=gridVectorN;		
	static int i=0;
	
	for(int processPointsI=0; processPointsI<processPointsCount; processPointsI++, i+=1) {
		if (i>=gridVectorN) {i=0;}
		VectorData& gridVec = gridVector[i];						
		float offsetFrac= useSpeed ? (VecLen(gridVec.v)/500) : (1.0/4000);
			
		unsigned ib = gridVec.indices>>16;
		unsigned jb = gridVec.indices&0xffff;
		Vec beg = Vec(gridSize*jb - gridCenter.x, 0.0f, gridSize*ib - gridCenter.z);		
		
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
		maxLength,
		useColor,
		useSpeed
		);	
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
	DrawString(-4,24," %d x %d", ps.nx, ps.ny);
		
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
		DrawString(0, 240,"time: %d", ps.curTime);

		fprintf(stdout, "centerX: %2.1f\n", cameraArrange.centerTranslation.x);
		fprintf(stdout, "centerY: %2.1f\n", cameraArrange.centerTranslation.y);
		fprintf(stdout, "centerZ: %2.1f\n", cameraArrange.centerTranslation.z);
		fprintf(stdout, "distance: %2.1f\n", cameraArrange.dist);
		fprintf(stdout, "rotLR: %2.1f\n", cameraArrange.rotLR);
		fprintf(stdout, "rotUD: %2.1f\n", cameraArrange.rotUD);
		fprintf(stdout, "exposure: %2.1f\n", ps.exposure);
		fprintf(stdout, "length: %2.1f\n", ps.length);
		fprintf(stdout, "intensity: %d\n", ps.pointCntExp);
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
							SDL_ShowCursor(0);
							noRedrawTimer.Reset();
							ps.sys.isActive=true;
						}
						break;
					case SDL_WINDOWEVENT_MINIMIZED:
						if (ps.sys.isActive){
							SDL_SetWindowGrab(display, SDL_FALSE);
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
	createdMp.resize(ps.nx*ps.ny*3);	
	separatedMp.resize(ps.nx*ps.ny*3*sizeof(MovingPoint)/sizeof(float));
	
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
	for (int i=0;i<ps.nx*ps.ny;i++)		
		sumLen += VecLen(gridVector[i].v);			
	float maxDotSpread = 1.0f/float(0.01*sumLen/(ps.nx*ps.ny));
	
	for (int i=0;i<ps.nx*ps.ny;i++)
		gridVector[i] = CreateGridVector(gridVector[i].v, gridVector[i].indices, maxDotSpread);
	
	//normalize total brightness
	double sumBrightness = 0;
	for (int i=0;i<ps.nx*ps.ny;i++) 
		sumBrightness += gridVector[i].dotBrightness;
	ps.totalBrightness = float(sumBrightness)/(screenW*screenW);
	
	if (!dumpFilename){
		//reduce initial exposure on small grids
		if ( sqrt(1.0*ps.nx*ps.ny)<700 )
			ps.exposure *= float(sqrt(1.0*ps.nx*ps.ny)/700);
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
			return 0;
		#endif		
		}
	else
		strcpy(inFileName,argv[1]);
		
	bool isOk=ReadInputFile(inFileName);
	if (!isOk) {
		fprintf(stdout,"Error opening input file: %s\n", inFileName);		
		return 1;
		}	



	if(CmdOptionExists(argv, argv + argc, "-c")){
		char *outFilename = GetCmdOption(argv, argv + argc, "-c");
		if(outFilename){
			bool isOk=ReadInputFile(inFileName);
			if (!isOk) {
				fprintf(stdout,"Error opening input file: %s\n", inFileName);
				return 1;
				}
			SaveVvfFile(outFilename);
			return 0;
		}
		else{
			fprintf(stdout,"Please specify output filename\n");
			return 1;
		}
	}

	char *centerX      = GetCmdOption(argv, argv + argc, "-centerX");
	char *centerY      = GetCmdOption(argv, argv + argc, "-centerY");
	char *centerZ      = GetCmdOption(argv, argv + argc, "-centerZ");
	char *rotLR        = GetCmdOption(argv, argv + argc, "-rotLR");
	char *rotUD        = GetCmdOption(argv, argv + argc, "-rotUD");
	char *distance     = GetCmdOption(argv, argv + argc, "-distance");
	char *exposure     = GetCmdOption(argv, argv + argc, "-exposure");
	char *length       = GetCmdOption(argv, argv + argc, "-length");
	char *intensity    = GetCmdOption(argv, argv + argc, "-intensity");
	char *time         = GetCmdOption(argv, argv + argc, "-time");
	char *dump         = GetCmdOption(argv, argv + argc, "-dump");

	showParams = CmdOptionExists(argv, argv + argc, "-params");
	offScreen  = CmdOptionExists(argv, argv + argc, "-offscreen");
	useColor = CmdOptionExists(argv, argv + argc, "-color");
	useSpeed = CmdOptionExists(argv, argv + argc, "-speed");

	if (offScreen){
		if (CmdOptionExists(argv, argv + argc, "-w") && CmdOptionExists(argv, argv + argc, "-h")){
			char *w = GetCmdOption(argv, argv + argc, "-w");
			char *h = GetCmdOption(argv, argv + argc, "-h");
			screenW = atoi(w);
			screenH = atoi(h);
		}
		else{
			fprintf(stdout,"Please specify render resolution using -w and -h\n");
			return 1;
		}
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


#if ( /*!defined(_DEBUG) && defined(WIN32)) || */ defined(FULLSCREEN) )
	if (!offScreen){
		SDL_DisplayMode current;
		SDL_GetCurrentDisplayMode(0, &current);
		screenW = current.w;
		screenH = current.h;
		display = SDL_CreateWindow(programName, 100, 100, screenW, screenH, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_INPUT_GRABBED);
#else
		display = SDL_CreateWindow(programName, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screenW, screenH, SDL_WINDOW_INPUT_GRABBED);
#endif
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
	SDL_ShowCursor(1);
	
	SDL_Quit();
	return 0;
}  
