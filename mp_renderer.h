#ifndef MP_RENDERER_H_INCLUDED
#define MP_RENDERER_H_INCLUDED

struct Vec
{
	float x,y,z;
	
	__host__ __device__ Vec(float px,float py,float pz) : x(px), y(py), z(pz) {}
	__host__ __device__ Vec(int px, int py, int pz) : x(float(px)), y(float(py)), z(float(pz)) {}
	__host__ __device__ Vec() {}
};

struct Camera
{
	Vec eye;
	Vec dir;   //forward direction, normalized
	Vec xd;    //right direction, normalized
	Vec yd;    //down direction, normalized
	Vec upLeftCornerTrans;
	float screenDist;   // distance to projection plane
};

struct SMpBufDesc
{
	int beg;
	int end;	
	int n;
	unsigned startTime;
};

struct PointProjection
{
	float x;
	float y;
	float zdistRec;
};


void CallMovingPointsRenderer(		
		Camera cam,
		float* mpData,
		SMpBufDesc* bufDesc,		
		int bufN,
		unsigned* intensityRaster,
		int screenW, int screenH,
		unsigned curtime,
		float brightnessMultiplier,
		float lengthMultiplier,
		float maxLength,
		bool useColor,
		bool useSpeed
		);

void SetOrtho(bool ortho);

#endif
