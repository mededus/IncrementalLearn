#include "StdAfx.h"
#include "classic_layer.h"

classic_layer::classic_layer(int t1, int t2, int t3)
{
	size = t1;
	connections = t2;
	this_planes = t3;
	//inicializacia mapy
	plane = new unit[this_planes];
	for ( int k = 0 ; k < this_planes ; k++) {
		plane[k].hkk = new float[connections];
		plane[k].hee = new float[connections];
		plane[k].w = new float[connections];
	}
	//nastavenie pociatocnych hodnot vah
	for ( int k = 0 ; k < this_planes ; k++) {
		for ( int i = 0 ; i < connections ; i++) {
				float weight = ((rand()%(int)(((10000) + 1)-(-10000)))+ (-10000));
				plane[k].w[i] = weight / (10000 / (2.4 / connections));
				plane[k].hkk[i] = 0.0;
				plane[k].hee[i] = 0.0;
			
		}
	//nastavenie 0 hodnot biasu a vystupov z neuronov
		plane[k].bias = 0.0;
		plane[k].biashkk = 0.0;
		plane[k].biashee = 0.0;
		plane[k].e = 0.0;
		plane[k].e2 = 0.0;
		plane[k].x = 0.0;
		plane[k].z = 0.0;
		plane[k].conf = 0.0;
		plane[k].membership = 0; 
	}
}

classic_layer::~classic_layer(void)
{
	for ( int a = 0 ; a < this_planes ; a++) {
		delete[] plane[a].hkk;
		delete[] plane[a].hee;
		delete[] plane[a].w;
	}
	delete[] plane;
}
