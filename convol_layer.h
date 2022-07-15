#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

class convol_layer			//trieda konvolucnej vrstvy
{
public:
	struct feature_map {	//struktura predstavujuca priznakovu mapu
		float **x;			//pole aktivacii neuronov v mape
		float **z;			//pole vstupov do neuronov v mape
		float **e;			//pole chyb v zmysle BP
		float **e2;			//pole estimacnych chyb v zmysle L-M
		float ***hkk;		//pomocna premenna L-M
		float ***hee;		//pomocna premenna L-M
		float ***w;			//vektor wah - konvolucny filter
		float bias;			//prah
		float biashkk;		//pomocna premenna L-M
		float biashee;		//pomocna premenna L-M
	} *plane;				//priznakove mapy
	int sizeM;				//rozmer mapy (matica m*n)
	int sizeN;
	int receptiveM;			//rozmer receptivneho pola (matica m*n)
	int receptiveN;
	int this_planes;		//pocet priznakovych map v tejto vrstve 
	int prev_planes;		//pocet priznakovych map v predchadzajucej vrstve
	short **connection_field;	
							//tabulka spojeni s predchadzajucou vrstvou 

	//sizeM, sizeN, receptiveM, receptiveN, this_planes, prev_planes, conection     
	convol_layer(int, int, int, int, int, int, int**);  
							//konstruktor
	~convol_layer(void);	//destruktor
};
