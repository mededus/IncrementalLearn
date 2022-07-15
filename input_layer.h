#pragma once

class input_layer
{
public:
	float **x;				//pole hodnot neuronov na vstupe
	int sizeM;				//rozmer M
	int sizeN;				//rozmer N

	input_layer(int, int);	//konstruktor
	~input_layer(void);		//destruktor
};
