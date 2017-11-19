#ifndef QUAD_RENDERER_H_INCLUDED
#define QUAD_RENDERER_H_INCLUDED

#include "mp_renderer.h"

struct SQuadBufDesc
{
	int gridWidth;
	int gridHeight;
	Vec vertices[4];
	int dataOffset;
};

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
	);

#endif
