#include "quad_renderer.h"

#define TILE_SIZE 3

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#else
#define HOST
#define DEVICE
#define GLOBAL
#include <vector_types.h>
#include <vector_functions.h>
#endif

namespace cuda_renderer
{

HOST DEVICE unsigned DivideCeil(unsigned dividend, unsigned divisor)
{
	return 1 + ((dividend - 1) / divisor);
}

DEVICE Vec VecCreate(float x, float y, float z);
DEVICE Vec VecAdd(Vec a, Vec b);
DEVICE Vec VecSub(Vec A, Vec B);
DEVICE float DotProduct(Vec A, Vec B);
DEVICE float VecLen(Vec a);
DEVICE Vec VecMul(Vec v,float t);
DEVICE Vec VecUnit(Vec a);
DEVICE Vec lerp(Vec v0, Vec v1, float t);
DEVICE PointProjection Proj(Vec t, Camera k, bool force);
DEVICE extern float rainbow[6][3];

DEVICE static float calculateSignedArea(const float2 tri[3])
{
	return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

DEVICE static float calculateBarycentricValue(float2 a, float2 b, float2 c, const float2 tri[3])
{
	float2 baryTri[3] = { a, b, c };
	return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

DEVICE static float3 calculateBarycentric(const float2 tri[3], float2 point)
{
	float beta = calculateBarycentricValue(make_float2(tri[0].x, tri[0].y), point, make_float2(tri[2].x, tri[2].y), tri);
	float gamma = calculateBarycentricValue(make_float2(tri[0].x, tri[0].y), make_float2(tri[1].x, tri[1].y), point, tri);
	float alpha = 1.0 - beta - gamma;
	return make_float3(alpha, beta, gamma);
}

DEVICE static bool isBarycentricCoordInBounds(const float3 barycentricCoord)
{
	return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
				 barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
				 barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

template <typename T> DEVICE static T combineBarycentric(float3 bary, T values[3])
{
	return values[0] * bary.x +
				 values[1] * bary.y +
				 values[2] * bary.z;
}

template <> DEVICE static float2 combineBarycentric(float3 bary, float2 values[3])
{
	return make_float2(
					values[0].x * bary.x +
					values[1].x * bary.y +
					values[2].x * bary.z,
					values[0].y * bary.x +
					values[1].y * bary.y +
					values[2].y * bary.z);
}

GLOBAL
void QuadRenderer(
	Camera cam,
	float *quadData,
	SQuadBufDesc *quadBufDesc,
	int numSlices,
	int numTiles,
	unsigned *intensityRaster,
	int screenW,
	int screenH,
	float brightnessMultiplier,
	float maxLength,
	float scale
	)
{
	unsigned blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	unsigned i = __mul24(blockId, blockDim.x) + threadIdx.x;

	const unsigned iSlice = i / numTiles;
	if (iSlice >= numSlices) return;

	SQuadBufDesc *pDesc = quadBufDesc + iSlice;

	PointProjection vertices[4] =
	{
		Proj(VecMul(pDesc->vertices[0], scale), cam, true),
		Proj(VecMul(pDesc->vertices[1], scale), cam, true),
		Proj(VecMul(pDesc->vertices[2], scale), cam, true),
		Proj(VecMul(pDesc->vertices[3], scale), cam, true),
	};

	if (vertices[0].zdistRec <= 0 && vertices[1].zdistRec <= 0 && vertices[2].zdistRec <= 0 && vertices[3].zdistRec <= 0) return;

	unsigned iTile = i % numTiles;

	struct
	{
		float2 min;
		float2 max;
	} boundingBox =
	{
		{
			fminf(fminf(vertices[0].x, vertices[1].x), fminf(vertices[2].x, vertices[3].x)),
			fminf(fminf(vertices[0].y, vertices[1].y), fminf(vertices[2].y, vertices[3].y)),
		},
		{
			fmaxf(fmaxf(vertices[0].x, vertices[1].x), fmaxf(vertices[2].x, vertices[3].x)),
			fmaxf(fmaxf(vertices[0].y, vertices[1].y), fmaxf(vertices[2].y, vertices[3].y)),
		},
	};

	unsigned bbWidth = min((unsigned)ceil(boundingBox.max.x), screenW) - max((unsigned)floor(boundingBox.min.x), 0);
	unsigned bbHeight = min((unsigned)ceil(boundingBox.max.y), screenH) - max((unsigned)floor(boundingBox.min.y), 0);
	unsigned tileX = iTile % DivideCeil(bbWidth, TILE_SIZE);
	unsigned tileY = iTile / DivideCeil(bbWidth, TILE_SIZE);
	if (tileY >= DivideCeil(bbHeight, TILE_SIZE)) return;

	int minX = max((int)round(boundingBox.min.x), 0) + tileX * TILE_SIZE;
	int minY = max((int)round(boundingBox.min.y), 0) + tileY * TILE_SIZE;
	int maxX = min(minX + TILE_SIZE, screenW);
	int maxY = min(minY + TILE_SIZE, screenH);

	float2 texCoords[] =
	{
		make_float2(0, pDesc->gridWidth * vertices[0].zdistRec),
		make_float2(0, 0),
		make_float2(pDesc->gridWidth * vertices[2].zdistRec, pDesc->gridHeight * vertices[2].zdistRec),
		make_float2(pDesc->gridWidth * vertices[3].zdistRec, 0),
	};

	for (int y = minY; y < maxY; ++y)
	{
		for (int x = minX; x < maxX; ++x)
		{
			float2 tri[] =
			{
				make_float2(vertices[0].x, vertices[0].y),
				make_float2(vertices[1].x, vertices[1].y),
				make_float2(vertices[2].x, vertices[2].y),
				make_float2(vertices[3].x, vertices[3].y),
			};
			float3 bary1 = calculateBarycentric(tri, make_float2(x, y));
			float3 bary2 = calculateBarycentric(tri + 1, make_float2(x, y));
			PointProjection *pVerts = nullptr;
			float2 *pTexCoords;
			if (isBarycentricCoordInBounds(bary1))
			{
				pVerts = vertices;
				pTexCoords = texCoords;
			}
			else if (isBarycentricCoordInBounds(bary2))
			{
				pVerts = vertices + 1;
				pTexCoords = texCoords + 1;
				bary1 = bary2;
			}

			if (pVerts)
			{
				float zs[] = { pVerts[0].zdistRec, pVerts[1].zdistRec, pVerts[2].zdistRec };
				float z = combineBarycentric(bary1, zs);
				if (z > 0)
				{
					int dstIndex = (x + y * screenW) * 3;
					float2 texCoord = combineBarycentric(bary1, pTexCoords);
					texCoord.x /= z;
					texCoord.y /= z;
					if (texCoord.x >= 0 && texCoord.y >= 0)
					{
						int texIndex = min((int)texCoord.y, pDesc->gridHeight) * pDesc->gridWidth + min((int)texCoord.x, pDesc->gridWidth);
						float length = quadData[pDesc->dataOffset + texIndex];
						float len = min(length / maxLength * 4, 4.f);
						int rainbowIndex = (int)len;
						float fade = len - rainbowIndex;
						Vec colorFrom = VecCreate(rainbow[rainbowIndex][0], rainbow[rainbowIndex][1], rainbow[rainbowIndex][2]);
						Vec colorTo = VecCreate(rainbow[rainbowIndex + 1][0], rainbow[rainbowIndex + 1][1], rainbow[rainbowIndex + 1][2]);
						Vec color = lerp(colorFrom, colorTo, fade);
						float brightness = length * length * 10 * z * z * brightnessMultiplier;

						intensityRaster[dstIndex]     += unsigned(brightness * color.x);
						intensityRaster[dstIndex + 1] += unsigned(brightness * color.y);
						intensityRaster[dstIndex + 2] += unsigned(brightness * color.z);
					}
				}
			}
		}
	}
}

}

void CallQuadRenderer(
	Camera cam,
	float *quadData,
	SQuadBufDesc *quadBufDesc,
	int numSlices,
	unsigned *intensityRaster,
	int screenW,
	int screenH,
	float brightnessMultiplier,
	float maxLength,
	float scale
	)
{
	if (numSlices < 1) return;

	unsigned numTiles = cuda_renderer::DivideCeil(screenW, TILE_SIZE) * cuda_renderer::DivideCeil(screenH, TILE_SIZE);
	unsigned numThreads = numTiles * numSlices;
	unsigned threadGroupSize = 128;
	dim3 threadGroups(cuda_renderer::DivideCeil(numThreads, threadGroupSize));
	while (threadGroups.x > 65535)
	{
		threadGroups.x /= 2;
		threadGroups.y *= 2;
	}

	cuda_renderer::QuadRenderer<<<threadGroups, threadGroupSize>>>(
		cam,
		quadData,
		quadBufDesc,
		numSlices,
		numTiles,
		intensityRaster,
		screenW,
		screenH,
		brightnessMultiplier,
		maxLength,
		scale
		);
}
