#include "StdAfx.h"
#include "input_layer.h"

input_layer::input_layer(int temp1, int temp2) 
{
	sizeM = temp1;
	sizeN = temp2;
	//sizeM je pocet riadkov, sizeN je pocet stlpcov
	//teda ide o klasicky zapis ako pri maticiach m*n
	//inicializacia vstupu
	x = new float*[sizeM];			
	for ( int i = 0 ; i < sizeM ; i++) {
		x[i] = new float[sizeN];
	}
	//nastavenie vsetkych hodnot pola na 0 
	for ( int i = 0 ; i < temp1 ; i++)
		for ( int j = 0 ; j < temp2 ; j++)
			x[i][j] = -0.1;
}

input_layer::~input_layer(void)
{
	for ( int i = 0 ; i < sizeM ; i++)
		delete[] x[i];
	delete[] x;
}
