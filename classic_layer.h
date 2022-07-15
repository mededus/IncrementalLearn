#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

class classic_layer		//trieda klasickej skrytej vrstvy
{
public:
	struct unit {		//struktura predstavujuca neuron
		float x;		//aktivacia neuronu
		float z;		//vstup do neuronu
		float conf;		//confidence neuronu
		float e;		//chyba v zmysle BP
		float e2;		//estimacna chyba v zmysle L-M
		float *hkk;		//pomocna premenna L-M
		float *hee;		//pomocna premenna L-M
		float *w;		//vektor vah veduci z predchadzajucej vrstvy
		float bias;		//prah
		float biashkk;	//pomocna premenna L-M
		float biashee;	//pomocna premenna L-M
		int membership; 
						//iba vystupna vrstva - znacka triedy
	} *plane;			//neurony
	int size;			//rozmer mapy = 1
	int connections;	//pocet spojeni s predch vrstvou
	int this_planes;	//pocet priznakovych map (neuronov)

	classic_layer(int, int, int);	//konstuktor
	~classic_layer(void);			//destruktor
};
