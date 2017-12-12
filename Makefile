


all:
	nvcc -O2 -o vector-viz `sdl2-config --libs` vector_viz_main.cpp fontMono_24.cpp renderer.cu `sdl2-config --cflags` -std=c++11

#nvcc -O2 -o vector-viz `sdl2-config --libs` vector_viz_main.cpp fontMono_24.cpp mp_renderer.cu `sdl2-config --cflags` -std=c++11
	
gtx:
	nvcc -O2 -o vector-viz -L/usr/local/lib -lSDL2 vector_viz_main.cpp fontMono_24.cpp renderer.cu -I/usr/local/include/SDL2 -D_REENTRANT -std=c++11

