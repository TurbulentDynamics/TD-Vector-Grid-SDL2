#include "mp_renderer.h"

const unsigned threadN=65536/4;

namespace cuda_renderer
{

__device__ 
Vec VecCreate(float x, float y, float z)
{
	Vec res;
	res.x=x;
	res.y=y;
	res.z=z;
	return res;
}

__device__ 
Vec VecAdd(Vec a, Vec b)
{
	return VecCreate(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ 
Vec VecSub(Vec A, Vec B) 
{
	A.x-=B.x;
	A.y-=B.y;
	A.z-=B.z;
	return A;	
}

__device__ 
float DotProduct(Vec A, Vec B) 
{ 
	return A.x*B.x + A.y*B.y + A.z*B.z;
}

__device__ 
float VecLen(Vec a) 
{ 
	return sqrt(DotProduct(a,a));
}
__device__ 
Vec VecMul(Vec v,float t)  
{ 
	v.x*=t;
	v.y*=t;
	v.z*=t;
	return v;	
}   
__device__ 
Vec VecUnit(Vec a) 
{ 
	return VecMul(a, 1.0f/VecLen(a));
}

__device__
Vec lerp(Vec v0, Vec v1, float t)
{
    return VecAdd(VecMul(v0, (1-t)), VecMul(v1, t));
}

__device__ float rainbow[][3] =
{
  { 1.0f, 0.0f, 0.0f },
  { 1.0f, 1.0f, 0.0f },
  { 0.0f, 1.0f, 0.0f },
  { 0.0f, 1.0f, 1.0f },
  { 0.0f, 0.0f, 1.0f },
  { 1.0f, 0.0f, 1.0f },
};



__device__
PointProjection PerspProj(Vec t, Camera k, bool force = false)
{	
	PointProjection ret;
	Vec diff=VecSub(t,k.eye);        
	float zdist = DotProduct(diff, k.dir);	
	
	if (!force && zdist < 0.1f) {
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

__global__ 
void MovingPointsRenderer(
		Camera cam,
		float* mpData,
		int mpN,
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
		)				
{		
	const unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;	
							
	int bufI=0;
	for (unsigned i=idx; true; i+=threadN)
		{		
		while (i>=bufDesc[bufI].n) {
			i -= bufDesc[bufI].n;
			bufI++;
			if (bufI>=bufN)
				return;
			}
		int pitch = bufDesc[bufI].n;				
		const unsigned timeMs = curtime - bufDesc[bufI].startTime;		
		float* data =&mpData[bufDesc[bufI].beg + i];
		
		float mpBegX = data[pitch*0];
		float mpBegY = data[pitch*1];
		float mpBegZ = data[pitch*2];
		float mpVelX = data[pitch*3];
		float mpVelY = data[pitch*4];
		float mpVelZ = data[pitch*5];
		float mpOffs = data[pitch*6];
		float mpBrig = data[pitch*7];
		
		Vec beg=VecCreate(mpBegX,mpBegY,mpBegZ);
		Vec v  =VecCreate(mpVelX,mpVelY,mpVelZ);
		//Vec v(1,1,1);
		float len = VecLen(v) ;
		float pos =mpOffs + timeMs * (useSpeed ? (len/500.0f) : (1.0/4000));
		if (pos>1) continue;		
		Vec p = VecAdd(beg,VecMul(v,pos*lengthMultiplier));
		
		PointProjection proj = PerspProj(p,cam);
		
		if (proj.zdistRec<=0) continue;
		float brightness = mpBrig * proj.zdistRec * proj.zdistRec * brightnessMultiplier;
				
		int x = int(proj.x);
		int y = int(proj.y);
		
		if (x<0 || x>=screenW) continue;
		if (y<0 || y>=screenH) continue;	
		int dstIndex = (x + y*screenW) * 3;
		len = min(len/ maxLength * 4, 4.f);
		int rainbowIndex = (int)len;
		float fade = len - rainbowIndex;
		Vec colorFrom = Vec(rainbow[rainbowIndex][0], rainbow[rainbowIndex][1], rainbow[rainbowIndex][2]);
		Vec colorTo = Vec(rainbow[rainbowIndex + 1][0], rainbow[rainbowIndex + 1][1], rainbow[rainbowIndex + 1][2]);
		Vec color = useColor ? lerp(colorFrom, colorTo, fade) : Vec(1, 1, 1);

		intensityRaster[dstIndex]     += unsigned(brightness * color.x);
		intensityRaster[dstIndex + 1] += unsigned(brightness * color.y);
		intensityRaster[dstIndex + 2] += unsigned(brightness * color.z);
		
		//intensityRaster[i%(100*1000)] += i;
		}
}

} //namespace cuda_renderer ends


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
		)				
{
	dim3 block(64);
	dim3 grid((unsigned int)ceil(threadN/(float)block.x));	
	
	if (bufN<1) return;
	
	cuda_renderer::MovingPointsRenderer<<<grid, block>>>(
		cam,
		mpData,
		0,
		bufDesc,		
		bufN,
		intensityRaster,
		screenW, screenH,
		curtime,
		brightnessMultiplier,
		lengthMultiplier,
		maxLength,
		useColor,
		useSpeed
		);
}


