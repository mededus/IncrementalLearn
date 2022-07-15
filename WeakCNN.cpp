#include "StdAfx.h"
#include "weakcnn.h"

WeakCNN::WeakCNN(int c, int s, int f, short af, short rb)
{
	p_INP = NULL;
	p_OUT = NULL;				
	p_vstup = NULL;
	info = NULL;
	classes = NULL;

	epsilon = 0.0;	
	composite_epsilon = 0.0;	
	beta = 1.0;		
	composite_beta = 1.0;		
	weight = 0.0;
	composite_weight = 0.0;
	weightN = 0.0;
	composite_weightN = 0.0;
	number_of_classes = 0;
	misclassification = 0;
	composite_misclassification = 0;
	dataset = -1;
	act_function = af;
	rbf_patterns = rb;
	filter_size = 0;
	filter_output = NULL;
	filter_heredity = 0;

	_CONVOLUTIONS = c;								
	p_C = new convol_layer*[c];
	for (int i = 0 ; i < c ; i++) 
		p_C[i] = NULL;
	
	_SAMPLINGS = s;
	p_S = new sampl_layer*[s];
	for (int i = 0 ; i < s ; i++) 
		p_S[i] = NULL;
	
	_CLASSICS = f;
	p_F = new classic_layer*[f];
	for (int i = 0 ; i < f ; i++) 
		p_F[i] = NULL;
}

WeakCNN::~WeakCNN(void)
{
	
	delete p_INP;
	delete p_OUT;
	delete p_vstup;
	delete info;
	delete classes;
	
	for (int i = 0 ; i < _CONVOLUTIONS ; i++) 
		delete p_C[i];
	delete p_C;

	for (int i = 0 ; i < _SAMPLINGS ; i++) 
		delete p_S[i];
	delete p_S;
	
	for (int i = 0 ; i < _CLASSICS ; i++) 
		delete p_F[i];
	delete p_F;
	
}

//******************************************************** UPRAVENE
// funkcia na inicializaciu novej siete pre ucenie
// return 2 - chyba pri otvoreni suboru
// return 1 - chyba pri alokacii pamati pre vrstvy - nie je mozne nacitat siet do vytvorenej struktury v triede
// return 0 - OK
int WeakCNN::initialize_net(char* name, bool initialize)
{
	FILE *def;
	int temp_int;										//pomocne premenne
	char temp_string[100];								//pomocne premenne
	int **temp_field;									//pomocne premenne
	int t1, t2, t11, t12, t21, t22, t3, t4, counter;	//pomocne premenne
	int CONVOLUTIONS = 0;			
	int SAMPLINGS = 0;				
	int CLASSICS = 0;


	if( !(def=fopen(name,"rt"))) {
		printf("\nError in opening file %s !", name);
		return(2);
	}
	else {
		//nastavenie topologie
		fscanf(def, "%s", &temp_string);
		fscanf(def, "\n%d %d %d", &CONVOLUTIONS, &SAMPLINGS, &CLASSICS);
		
		if (CONVOLUTIONS != _CONVOLUTIONS || SAMPLINGS != _SAMPLINGS || CLASSICS != _CLASSICS)
			return(1);

		info = new char[_CONVOLUTIONS+_SAMPLINGS+_CLASSICS+1];
		counter = 0;
		
		if (_CONVOLUTIONS != _SAMPLINGS)
			return(1);

		fscanf(def, "%s", &temp_string);
		fscanf(def, "\n%d %d", &t1, &t2);
			p_INP = new input_layer(t1, t2);
			t4 = 1;
		
		for (int a = 0 ; a < _CONVOLUTIONS ; a++) {
			fscanf(def, "\n%s", &temp_string);
			fscanf(def, "\n%d %d %d %d %d", &t3, &t11, &t12, &t21, &t22);
			fscanf(def, "\n%s", &temp_string);
				temp_field = new int*[t3];				
				for (int i = 0 ; i < t3 ; i++) 
					temp_field[i] = new int[t4];
				for (int i = 0 ; i < t3 ; i++) {
					for (int j = 0 ; j < t4 ; j++) {
						fscanf(def, "%d", &temp_field[i][j]);
					}
				}
				p_C[a] = new convol_layer(t11,t12, t21, t22, t3, t4, temp_field);
				info[counter] = 'c';
				counter++;
				for (int i = 0 ; i < t3 ; i++)
					delete[] temp_field[i];
				delete[] temp_field;
			fscanf(def, "\n%s", &temp_string);
			fscanf(def, "\n%d %d %d %d %d", &t3, &t11, &t12, &t21, &t22);
				p_S[a] = new sampl_layer(t11, t12, t21, t22, t3);
				info[counter] = 's';
				counter++;
				t4 = t3;
		}
		for (int a = 0 ; a < _CLASSICS ; a++) {
			fscanf(def, "\n%s", &temp_string);
			fscanf(def, "\n%d %d %d", &t3, &t1, &t2);
				if (a == 0) {
					filter_size = t2;
					filter_output = new float[t2];
				}
				p_F[a] = new classic_layer(t1,t2,t3);
				info[counter] = 'f';
				counter++;
		}
		fscanf(def, "\n%s", &temp_string);
		fscanf(def, "\n%d %d %d", &t3, &t1, &t2);
			p_OUT = new classic_layer (t1,t2,t3);
			info[counter] = 'o';
			counter++;
		
		classes = new int[t3];
		number_of_classes = t3;
		
		fscanf(def, "\n%s", &temp_string);
		fscanf(def, "\n");
		for (int a = 0 ; a < t3 ; a++) {
			fscanf(def, "%d", &temp_int);
			p_OUT->plane[a].membership = classes[a] = temp_int;

		}
		
		fscanf(def, "\n%s", &temp_string);
		fscanf(def, "%d", &dataset);	
		fscanf(def, "%lf", &epsilon);
		fscanf(def, "%lf", &beta);
		fscanf(def, "%lf", &weight);
		fscanf(def, "%lf", &composite_epsilon);
		fscanf(def, "%lf", &composite_beta);
		fscanf(def, "%lf", &composite_weight);

		//ak je initialize = false, tak sa vahy nastavia podla suboru
		if (initialize == false) {
			int convol = 0;
			int sampl = 0;
			int classic = 0;
			float value;
			for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS+1 ; a++) {
				switch (info[a]) {
					case 'c':
					if (convol > _CONVOLUTIONS)
						exit(0);
					fscanf(def, "\n%s", &temp_string);
					for (int k = 0 ; k < p_C[convol]->this_planes ; k++) {
						fscanf(def, "%f", &value);	//bias
						p_C[convol]->plane[k].bias = value;
						int max;
						if (convol == 0)
							max = 1;	//bereme zo vstupu
						else
							max = p_S[sampl-1]->this_planes;	//z predch. vrstvy
						for (int h = 0 ; h < max ; h++) {
							for (int i = 0 ; i < p_C[convol]->receptiveM ; i++) {
								for (int j = 0 ; j < p_C[convol]->receptiveN ; j++) {
									fscanf(def, "%f", &value);
									p_C[convol]->plane[k].w[h][i][j] = value;
								}
							}
						}
					}
					convol++;
					break;
					case 's':
					if (sampl > _SAMPLINGS)
						exit(0);
					fscanf(def, "\n%s", &temp_string);
					for (int k = 0 ; k < p_S[sampl]->this_planes ; k++) {
						fscanf(def, "%f", &value);	//bias
						p_S[sampl]->plane[k].bias = value;
						fscanf(def, "%f", &value);
						p_S[sampl]->plane[k].w = value;
					}
					sampl++;
					break;
					case 'f':
					if (classic > _CLASSICS)
						exit(0);
					fscanf(def, "\n%s", &temp_string);
					for (int k = 0 ; k < p_F[classic]->this_planes ; k++) {
						fscanf(def, "%f", &value);	//bias
						p_F[classic]->plane[k].bias = value;
						for (int h = 0 ; h < p_F[classic]->connections ; h++) {
							fscanf(def, "%f", &value);
							p_F[classic]->plane[k].w[h] = value;
						}
					}
					classic++;
					break;
					case 'o':
					fscanf(def, "\n%s", &temp_string);
					for (int k = 0 ; k < p_OUT->this_planes ; k++) {
						fscanf(def, "%f", &value);	//bias
						p_OUT->plane[k].bias = value;
						for (int h = 0 ; h < p_OUT->connections ; h++) {
							fscanf(def, "%f", &value);
							p_OUT->plane[k].w[h] = value;
						}
					}
					break;
					
				}
			}
		}
		return(0);
	}
}


int WeakCNN::reset_net(void)
{
	
	
	return 0;
}

//********************************************************  UPRAVENE
// funkcia na ulozenie siete do suboru
// return 0 - OK
// return 1 - chyba
int WeakCNN::save_net(char* name)
{
	FILE *f;
	
	if( !(f=fopen(name,"wt"))) {
		printf("\nError in saving file !");
		return(1);
	}
	else {
		//hlavicka
		fprintf(f, "#layers");
		fprintf(f, "\n%d %d %d", _CONVOLUTIONS, _SAMPLINGS, _CLASSICS);
		fprintf(f, "\n#input");
		fprintf(f, "\n%d %d", p_INP->sizeM, p_INP->sizeN);
		int convol = 0;
		int sampl = 0;
		int classic = 0;
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS+1 ; a++) {
			switch (info[a]) {
				case 'c':
				case 'd':
				case 'e':
				fprintf(f, "\n#convolution");
				fprintf(f, "\n%d %d %d %d %d", p_C[convol]->this_planes, p_C[convol]->sizeM, p_C[convol]->sizeN, p_C[convol]->receptiveM, p_C[convol]->receptiveN);
				fprintf(f, "\n#connection\n");
				for (int i = 0 ; i < p_C[convol]->this_planes ; i++) {
					for (int j = 0 ; j < p_C[convol]->prev_planes ; j++) {
						fprintf(f, "%d ", p_C[convol]->connection_field[i][j]);
					}
					fprintf(f, "\n");
				}
				convol++;
				break;
				case 's':
				case 't':
				case 'u':
				fprintf(f, "#sampling");
				fprintf(f, "\n%d %d %d %d %d", p_S[sampl]->this_planes, p_S[sampl]->sizeM, p_S[sampl]->sizeN, p_S[sampl]->receptiveM, p_S[sampl]->receptiveN);
				sampl++;
				break;
				case 'f':
				fprintf(f, "\n#classic");
				fprintf(f, "\n%d %d %d", p_F[classic]->this_planes, p_F[classic]->size, p_F[classic]->connections);
				classic++;
				break;
				case 'o':
				fprintf(f, "\n#output");
				fprintf(f, "\n%d %d %d", p_OUT->this_planes, p_OUT->size, p_OUT->connections);
				classic++;
				break;
			
			}
		}
		fprintf(f, "\n#membership");
		fprintf(f, "\n");
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			if (i == 0)
				fprintf(f, "%d", p_OUT->plane[i].membership); 
			else
				fprintf(f, " %d", p_OUT->plane[i].membership); 
		}
		
		fprintf(f, "\n#parameters");
		fprintf(f, "\n%d", dataset);
		fprintf(f, " %lf", epsilon);
		fprintf(f, " %lf", beta);
		fprintf(f, " %lf", weight);
		fprintf(f, " %lf", composite_epsilon);
		fprintf(f, " %lf", composite_beta);
		fprintf(f, " %lf", composite_weight);
		
		convol = 0;
		sampl = 0;
		classic = 0;
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS+1 ; a++) {
			fprintf(f, "\n#weights_in_layer_%d", a+1);
			switch (info[a]) {
				case 'c':
				case 'd':
				case 'e':
				for (int k = 0 ; k < p_C[convol]->this_planes ; k++) {
					fprintf(f, "\n%f", p_C[convol]->plane[k].bias);	
					int max;
					if (convol == 0)
						max = 1;	//bereme zo vstupu
					else
						max = p_S[sampl-1]->this_planes;	//z predch. vrstvy
					for (int h = 0 ; h < max ; h++) {
						for (int i = 0 ; i < p_C[convol]->receptiveM ; i++) {
							for (int j = 0 ; j < p_C[convol]->receptiveN ; j++) {
								fprintf(f, " %f", p_C[convol]->plane[k].w[h][i][j]);
							}
						}
					}
				}
				convol++;
				break;
				case 's':
				case 't':
				case 'u':
				for (int k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					fprintf(f, "\n%f", p_S[sampl]->plane[k].bias);
					fprintf(f, " %f", p_S[sampl]->plane[k].w);
				}
				sampl++;
				break;
				case 'f':
				for (int k = 0 ; k < p_F[classic]->this_planes ; k++) {
					fprintf(f, "\n%f", p_F[classic]->plane[k].bias);
					for (int h = 0 ; h < p_F[classic]->connections ; h++) {
						fprintf(f, " %f", p_F[classic]->plane[k].w[h]);
					}
				}
				classic++;
				break;
				case 'o':
				for (int k = 0 ; k < p_OUT->this_planes ; k++) {
					fprintf(f, "\n%f", p_OUT->plane[k].bias);	
					for (int h = 0 ; h < p_OUT->connections ; h++) {
						fprintf(f, " %f", p_OUT->plane[k].w[h]);
					}
				}
				break;
			}
		}
		fclose(f);
		printf("\nNetwork saved (%s)", name);
		return(0);
	}
}

//********************************************************
// prechodova funkcia 
float WeakCNN::squash_f(float value)
{
	
	if (act_function == 0) {
		//value *= (float)2.0/3.0;
		//return (1.7159 * (exp(2*value)-1.0) / (exp(2*value)+1.0));
		return (1.7159 * (exp(value*4.0/3.0)-1.0) / (exp(value*4.0/3.0)+1.0));
		//return (1.7159*tanh((2.0/3.0)*value));

	}
	else {
		return (1.0 / (1.0 + exp(-value)));
	}
}

//********************************************************
// derivacia prechodovej funkcie 
float WeakCNN::squash_df(float value)
{
	
	if (act_function == 0) {
		//return ((1.7159 * 4.0) / (exp(2*value) + 2.0 + exp(-2*value)));
		//return (1.7159-value*value/1.7159)*2.0/3.0;
		
		return (4.5757 / (exp(value*4.0/3.0) + 2.0 + exp(-value*4.0/3.0)));
	}
	else {
		return (exp(-value) / ((1.0 + exp(-value)) * (1.0 + exp(-value))));
	}
}

//********************************************************
// druha derivacia prechodovej funkcie 
float WeakCNN::squash_ddf(float value)
{
	if (act_function == 0) {
		float a = 2.0/3.0;
		return (9.1515*((exp(-2*a*value)-exp(2*a*value))/(pow((exp(a*value)+exp(-a*value)),4))));
	}
	else {
		return ((exp(-3*value)-exp(-value))/(pow((1.0+exp(-value)),4)));
	}
}

//******************************************************** UPRAVENE dokoncit
// funkcia na prechod sietou
void WeakCNN::feed_forward(void)
{
	int k,i,j,m,n,h,tmp;
	int convol = 0;
	int sampl = 0;
	int classic = 0;
	float value;
	if (filter_heredity == 0) {
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
			switch (info[a]) {
				case 'c':
				for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
					for (i = 0 ; i < p_C[convol]->sizeM ; i++) {
						for (j = 0 ; j < p_C[convol]->sizeN ; j++) {
							//vypocet hodnoty pre neuron vo vrstve C3
							float value = 0;
							//plany su pospajane podla connection_field
							for (h = 0 ; h < p_C[convol]->prev_planes ; h++) {
								int xx = 0;
								int yy = 0;
								for (m = i ; m < i + p_C[convol]->receptiveM /*- 1*/ ; m++) {
									for (n = j ; n < j + p_C[convol]->receptiveN /*- 1*/ ; n++) {
										if (convol == 0)
											value += p_INP->x[m][n]*p_C[convol]->plane[k].w[h][xx][yy];
										else
											value += p_S[sampl-1]->plane[h].x[m][n]*p_C[convol]->plane[k].w[h][xx][yy];
										yy++;
									}
									xx++;
									yy = 0;
								}
							}
							value += p_C[convol]->plane[k].bias;//*(-1);
							p_C[convol]->plane[k].z[i][j] = value;
							p_C[convol]->plane[k].x[i][j] = squash_f(value);
						}
					}
				}
				convol++;
				break;
				case 'd':
				for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
					for (i = 0 ; i < p_C[convol]->sizeM ; i++) {
						for (j = 0 ; j < p_C[convol]->sizeN ; j++) {
							//vypocet hodnoty pre neuron vo vrstve C3
							float value = 0;
							//plany su pospajane podla connection_field
							for (h = 0 ; h < p_C[convol]->prev_planes ; h++) {
								int xx = 0;
								int yy = 0;
								for (m = i ; m < i + p_C[convol]->receptiveM /*- 1*/ ; m++) {
									for (n = j ; n < j + p_C[convol]->receptiveN /*- 1*/ ; n++) {
										if (convol == 0)
											value += p_INP->x[m][n]*p_C[convol]->plane[k].w[h][xx][yy];
										else
											value += p_S[sampl-1]->plane[h].x[m][n]*p_C[convol]->plane[k].w[h][xx][yy];
										yy++;
									}
									xx++;
									yy = 0;
								}
							}
							//value += p_C[convol]->plane[k].bias*(-1);
							p_C[convol]->plane[k].z[i][j] = value;
							p_C[convol]->plane[k].x[i][j] = squash_f(value);
						}
					}
				}
				convol++;
				break;
				case 's':
				tmp = 0;
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							//vypocet hodnoty pre neuron vo vrstve S2
							float value = 0;
							int recM = p_S[sampl]->receptiveM;
							int recN = p_S[sampl]->receptiveN;
							for (m = i*recM ; m < i*recM + recM ; m++) {              // asi treba opravit
								for (n = j*recN ; n < j*recN + recN ; n++) {
									value += p_C[convol-1]->plane[k].x[m][n];
								}
							}
							value *= p_S[sampl]->plane[k].w;
							value += p_S[sampl]->plane[k].bias;//*(-1);
							p_S[sampl]->plane[k].z[i][j] = value;
							p_S[sampl]->plane[k].x[i][j] = squash_f(value);

							if (sampl == _SAMPLINGS-1) {
								filter_output[tmp] = p_S[sampl]->plane[k].x[i][j];
							}
							tmp++;
						}
					}
				}

				sampl++;
				break;
				case 't':
				tmp = 0;
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							//vypocet hodnoty pre neuron vo vrstve S2
							float value = 0;
							int recM = p_S[sampl]->receptiveM;
							int recN = p_S[sampl]->receptiveN;
							for (m = i*recM ; m < i*recM + recM ; m++) {              // asi treba opravit
								for (n = j*recN ; n < j*recN + recN ; n++) {
									value += p_C[convol-1]->plane[k].x[m][n];
								}
							}
							value *= p_S[sampl]->plane[k].w;
							//value += p_S[sampl]->plane[k].bias*(-1);
							p_S[sampl]->plane[k].z[i][j] = value / (recM * recN);
							p_S[sampl]->plane[k].x[i][j] = squash_f(value);

							if (sampl == _SAMPLINGS-1) {
								filter_output[tmp] = p_S[sampl]->plane[k].x[i][j];
							}
							tmp++;
						}
					}
				}
				sampl++;
				break;
				case 'f':
				for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
					float value = 0;
					int temp = 0;
					if (classic == 0) {
						for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
							for (i = 0 ; i < p_S[sampl-1]->sizeM ; i++) {
								for (j = 0 ; j < p_S[sampl-1]->sizeN ; j++) {
									value += p_S[sampl-1]->plane[h].x[i][j]*p_F[classic]->plane[k].w[temp];
									temp++;
								}
							}
						}
					}
					else {
						for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
							value += p_F[classic-1]->plane[h].x*p_F[classic]->plane[k].w[temp];
							temp++;
						}
					}
					value += p_F[classic]->plane[k].bias;//*(-1);
					p_F[classic]->plane[k].z = value;
					p_F[classic]->plane[k].x = squash_f(value);

				}
				classic++;
				break;
			}
		}
	}
	else {
		for (int classic = 0 ; classic < _CLASSICS ; classic++) {
			for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
				float value = 0;
				int temp = 0;
				if (classic == 0) {
					for (h = 0 ; h < filter_size ; h++) {
						value += filter_output[h]*p_F[classic]->plane[k].w[temp];
						temp++;
					}
				}
				else {
					for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
						value += p_F[classic-1]->plane[h].x*p_F[classic]->plane[k].w[temp];
						temp++;
					}
				}
				value += p_F[classic]->plane[k].bias;//*(-1);
				p_F[classic]->plane[k].z = value;
				p_F[classic]->plane[k].x = squash_f(value);

			}

		}

	}
	
	if (rbf_patterns == 0) {
		for (k = 0 ; k < p_OUT->this_planes ; k++) {
			float value = 0;
			int temp = 0;
			for (h = 0 ; h < p_F[_CLASSICS-1]->this_planes ; h++) {
				value += p_F[_CLASSICS-1]->plane[h].x*p_OUT->plane[k].w[temp];
				temp++;
			}
			value += p_OUT->plane[k].bias;//*(-1);
			p_OUT->plane[k].z = value;
			p_OUT->plane[k].x = squash_f(value);
		}
	}
}

//******************************************************** UPRAVENE dokoncit
// funkcia algoritmu spatneho sirenia chyby - 1.derivacia
void WeakCNN::back_propagation_e(int membership, rbfpatterns *rbfPAT, float &output_error, float &bpe_error)
{
	int i,j,k,m,n,h,s;
	int convol = _CONVOLUTIONS-1;
	int sampl = _SAMPLINGS-1;
	int classic = _CLASSICS-1;
	float output_e = 0.0;
	float bpe_e = 0.0;
	
	if (rbf_patterns == 0) {
		//aktivacna funkcia tanh
		if (act_function == 0) {
			for (int a = 0 ; a < p_OUT->this_planes ; a++) {
				if (p_OUT->plane[a].membership == membership) {
					//p_OUT->plane[a].e = ( + p_OUT->plane[a].x - 1.0)*squash_df(p_OUT->plane[a].z);
					//output_e += 0.5 * pow( + p_OUT->plane[a].x - 1.0, 2);
					p_OUT->plane[a].e = ( 1.0 - p_OUT->plane[a].x)*squash_df(p_OUT->plane[a].z);
					output_e += 0.5 * pow( 1.0 - p_OUT->plane[a].x, 2);

				}
				else {
					//p_OUT->plane[a].e = ( + p_OUT->plane[a].x + 1.0)*squash_df(p_OUT->plane[a].z);
					//output_e += 0.5 * pow( + p_OUT->plane[a].x + 1.0, 2);
					p_OUT->plane[a].e = ( -1.0 - p_OUT->plane[a].x)*squash_df(p_OUT->plane[a].z);
					output_e += 0.5 * pow( -1.0 - p_OUT->plane[a].x, 2);

				}
				//sum_error += fabs(p_OUT->plane[a].e);
				//bpe_e += pow(p_OUT->plane[a].e,2);
				bpe_e += fabs(p_OUT->plane[a].e);
			}
		}
		//aktivacna funckia sigmodia
		else {
			for (int a = 0 ; a < p_OUT->this_planes ; a++) {
				if (p_OUT->plane[a].membership == membership) {
					p_OUT->plane[a].e = ( 1.0 - p_OUT->plane[a].x)*squash_df(p_OUT->plane[a].z);
					output_e += 0.5 * pow( 1.0 - p_OUT->plane[a].x, 2);
				}
				else {
					p_OUT->plane[a].e = ( 0.0 - p_OUT->plane[a].x)*squash_df(p_OUT->plane[a].z);
					output_e += 0.5 * pow( 0.0 - p_OUT->plane[a].x, 2);
				}
				//sum_error += fabs(p_OUT->plane[a].e);
				//bpe_e += pow(p_OUT->plane[a].e,2);
				bpe_e += fabs(p_OUT->plane[a].e);
			}

		}
		//bpe_e /= p_OUT->this_planes;
		output_error = output_e;
		bpe_error = bpe_e;
	}

	for (int a = _CONVOLUTIONS+_SAMPLINGS+_CLASSICS-1 ; a >= 0 ; a--) {
		switch (info[a]) {
			case 'c':
			
			for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
				h = k;
				for (i = 0 ; i < p_C[convol]->sizeM ; i++) {
					for (j = 0 ; j < p_C[convol]->sizeN ; j++) {           // je treba opravit             
						float E = p_S[sampl+1]->plane[h].w*p_S[sampl+1]->plane[h].e[i/p_S[sampl+1]->receptiveM][j/p_S[sampl+1]->receptiveN];
						p_C[convol]->plane[k].e[i][j] = E*squash_df(p_C[convol]->plane[k].z[i][j]);
						
					}
				}
			}
			
			convol--;
			break;
			case 'd':
			convol--;
			break;
			case 'e':
			convol--;
			break;
			case 's':
			
			if (sampl == _SAMPLINGS-1) {	//posledna vzorkovacia
				int temp = 0;
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							float E = 0.0;
							for (h = 0 ; h < p_F[0]->this_planes ; h++) {
								E +=p_F[0]->plane[h].w[temp]*p_F[0]->plane[h].e;
							}
							p_S[sampl]->plane[k].e[i][j] = E*squash_df(p_S[sampl]->plane[k].z[i][j]);
							temp++;
						}
					}
				}
			}
			else {
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							//kazdy neuron moze byt spojeny az z m*n neuronmi s prislusnym planom C3 
							float E = 0.0;
							int xx,yy;
							for (h = 0 ; h < p_C[convol+1]->this_planes ; h++) {
								for (m = 0 ; m < p_C[convol+1]->receptiveM ; m++) {
									for (n = 0 ; n < p_C[convol+1]->receptiveN ; n++) {
										xx = i - m;
										yy = j - n;
										if ((xx >= 0 && xx < p_C[convol+1]->sizeM) && (yy >=0 && yy < p_C[convol+1]->sizeN)) {  
											E +=p_C[convol+1]->plane[h].w[k][m][n]*p_C[convol+1]->plane[h].e[xx][yy];
										}
									}
								}
							}
							p_S[sampl]->plane[k].e[i][j] = E*squash_df(p_S[sampl]->plane[k].z[i][j]);
						}
					}
				}
			}
			
			sampl--;
			break;
			case 't':
			sampl--;
			break;
			case 'u':
			sampl--;
			break;
			case 'f':	
			if (classic == _CLASSICS-1) {	//predposledna vrstva siete
				
				if (rbf_patterns == 0) {			
					for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
						float E = 0.0;
						for (h = 0 ; h < p_OUT->this_planes ; h++) {
							E += p_OUT->plane[h].w[k]*p_OUT->plane[h].e;
						}
						p_F[classic]->plane[k].e = E*squash_df(p_F[classic]->plane[k].z);
					}
				}
				else {
					
					i = 0;
					j = 0;
					for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
						p_F[classic]->plane[k].e = (rbfPAT->pattern[membership].pixel[i][j] - p_F[classic]->plane[k].x)*squash_df(p_F[classic]->plane[k].z);
						output_e += 0.5 * pow(rbfPAT->pattern[membership].pixel[i][j] - p_F[classic]->plane[k].x, 2);
						bpe_e += fabs(p_F[classic]->plane[k].e);
						j++;
						if (j == rbfPAT->size_y) {
							j = 0;
							i++;
						}	
					}
					
					output_error = output_e;
					bpe_error = bpe_e;
				}
			}
			else {
				for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
					float E = 0.0;
					for (h = 0 ; h < p_F[classic+1]->this_planes ; h++) {
						E += p_F[classic+1]->plane[h].w[k]*p_F[classic+1]->plane[h].e;
					}
					p_F[classic]->plane[k].e = E*squash_df(p_F[classic]->plane[k].z);
				}
			}
			classic--;
			break;
		}
	}
}

//******************************************************** UPRAVENE dokoncit
// funkcia algoritmu spatneho sirenia chyby - 2.derivacia
void WeakCNN::back_propagation_e2(int membership)
{
	int i,j,k,m,n,h;
	int convol = _CONVOLUTIONS-1;
	int sampl = _SAMPLINGS-1;
	int classic = _CLASSICS-1;
	float sum_error = 0.0;
	float A = 1.7159;
	
	// !!! otestovat ci OK
	if (rbf_patterns == 0) {
		for (int a = 0 ; a < p_OUT->this_planes ; a++) {

			p_OUT->plane[a].e2 = 1.0;
		}
	}

	for (int a = _CONVOLUTIONS+_SAMPLINGS+_CLASSICS-1 ; a >= 0 ; a--) {
		switch (info[a]) {
			case 'c':
			
			for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
				h = k;
				for (i = 0 ; i < p_C[convol]->sizeM ; i++) {
					for (j = 0 ; j < p_C[convol]->sizeN ; j++) {
						//float E2 = p_S[sampl+1]->plane[h].w*(A-p_C[convol]->plane[k].x[i][j]*p_C[convol]->plane[k].x[i][j]/A)*4/9;
						//p_C[convol]->plane[k].e2[i][j] = E2*E2*p_S[sampl+1]->plane[h].e2[i/p_S[sampl+1]->receptiveM][j/p_S[sampl+1]->receptiveN];  //niesom isty
						
						//float E2 = pow(p_S4->plane[h].w,2)*p_S4->plane[h].e2[i/p_S[sampl+1]->receptive][j/p_S[sampl+1]->receptive];
						//p_C3->plane[k].e2[i][j] = E2*squash_df(pow(p_C3->plane[k].x[i][j],2),false);

						float E2 = p_S[sampl+1]->plane[h].w*p_S[sampl+1]->plane[h].w*p_S[sampl+1]->plane[h].e2[i/p_S[sampl+1]->receptiveM][j/p_S[sampl+1]->receptiveN];
						p_C[convol]->plane[k].e2[i][j] = E2*squash_df(p_C[convol]->plane[k].z[i][j])*squash_df(p_C[convol]->plane[k].z[i][j]);
					}
				}
			}
		
			convol--;
			break;
			case 'd':
			convol--;
			break;
			case 'e':
			convol--;
			break;
			case 's':
			
			if (sampl == _SAMPLINGS-1) {	//posledna vzorkovacia
				int temp = 0;
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							float E2 = 0.0;
							for (h = 0 ; h < p_F[0]->this_planes ; h++) {
								E2 += p_F[0]->plane[h].w[temp]*p_F[0]->plane[h].w[temp]*p_F[0]->plane[h].e2;
							}
							//p_S[sampl]->plane[k].e2[i][j] = E2*(A-p_S[sampl]->plane[k].x[i][j]*p_S[sampl]->plane[k].x[i][j]/A)*(A-p_S[sampl]->plane[k].x[i][j]*p_S[sampl]->plane[k].x[i][j]/A)*4/9;
							p_S[sampl]->plane[k].e2[i][j] = E2*squash_df(p_S[sampl]->plane[k].z[i][j])*squash_df(p_S[sampl]->plane[k].z[i][j]);
							temp++;
						}
					}
				}
			}
			else {
				for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
					for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
						for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
							float E2 = 0.0;
							int xx,yy;
							for (h = 0 ; h < p_C[convol+1]->this_planes ; h++) {
								for (m = 0 ; m < p_C[convol+1]->receptiveM ; m++) {
									for (n = 0 ; n < p_C[convol+1]->receptiveN ; n++) {
										xx = i - m;
										yy = j - n;
										if ((xx >= 0 && xx < p_C[convol+1]->sizeM) && (yy >=0 && yy < p_C[convol+1]->sizeN)) {  
											E2 += p_C[convol+1]->plane[h].w[k][m][n]*p_C[convol+1]->plane[h].w[k][m][n]*p_C[convol+1]->plane[h].e2[xx][yy];
										}
									}
								}
							}
							//p_S[sampl]->plane[k].e2[i][j] = E2*(A-p_S[sampl]->plane[k].x[i][j]*p_S[sampl]->plane[k].x[i][j])*(1-p_S[sampl]->plane[k].x[i][j]*p_S[sampl]->plane[k].x[i][j]/A)*4/9;
							p_S[sampl]->plane[k].e2[i][j] = E2*squash_df(p_S[sampl]->plane[k].z[i][j])*squash_df(p_S[sampl]->plane[k].z[i][j]);
						}
					}
				}
			}
			
			sampl--;
			break;
			case 't':
			sampl--;
			break;
			case 'u':
			sampl--;
			break;
			case 'f':	
			if (classic == _CLASSICS-1) {	//predposledna vrstva siete
			
				if (rbf_patterns == 0) {
					for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
						float E2 = 0.0;
						for (h = 0 ; h < p_OUT->this_planes ; h++) {
							E2 += p_OUT->plane[h].w[k]*p_OUT->plane[h].w[k]*p_OUT->plane[h].e2;
						}
						//p_F[classic]->plane[k].e2 = E2*(A-p_F[classic]->plane[k].x*p_F[classic]->plane[k].x/A)*(A-p_F[classic]->plane[k].x*p_F[classic]->plane[k].x/A)*4/9;
						p_F[classic]->plane[k].e2 = E2*squash_df(p_F[classic]->plane[k].z)*squash_df(p_F[classic]->plane[k].z);
					}
				}
				else {
					for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
						p_F[classic]->plane[k].e2 = 1.0;
					}
				}
	
			}
			else {
				for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
					float E2 = 0.0;
					for (h = 0 ; h < p_F[classic+1]->this_planes ; h++) {
						E2 += p_F[classic+1]->plane[h].w[k]*p_F[classic+1]->plane[h].w[k]*p_F[classic+1]->plane[h].e2;
					}
					//p_F[classic]->plane[k].e2 = E2*(A-p_F[classic]->plane[k].x*p_F[classic]->plane[k].x/A)*(A-p_F[classic]->plane[k].x*p_F[classic]->plane[k].x/A)*4/9;
					p_F[classic]->plane[k].e2 = E2*squash_df(p_F[classic]->plane[k].z)*squash_df(p_F[classic]->plane[k].z);
				}
			}
			classic--;
			break;
		}
	}
}

//******************************************************* UPRAVENE dokoncit
// funkcia nastavenia hodnot novych vah podla chyby
void WeakCNN::adjust_weights(float ETA, float MI)
{
	int k,i,j,m,n,h;
	int convol = 0;
	int sampl = 0;
	int classic = 0;
	float GAMA = 0;			//vysledny uciaci koef.
	if (filter_heredity == 0) {
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
			switch (info[a]) {
				case 'c':
				
					for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
						float error = 0.0;
						int counter = 0;
						for (i = 0 ; i < p_C[convol]->sizeM ; i++) {
							for (j = 0 ; j < p_C[convol]->sizeN ; j++) {
								error += p_C[convol]->plane[k].e[i][j];
								counter++;
							}
						}
						p_C[convol]->plane[k].bias += (ETA/(MI+p_C[convol]->plane[k].biashkk))*error;//counter;
						//p_C[convol]->plane[k].bias -= (ETA/MI)*(-1)*error;//counter;
						//counter = 0;
						for (i = 0 ; i < p_C[convol]->receptiveM ; i++) {
							for (j = 0 ; j < p_C[convol]->receptiveN ; j++) {
								float value = 0.0;
								counter = 0.0;
								int xx = 0;
								int yy = 0;
								if (convol == 0) {		//bereme zo vstupu
									xx = 0;
									yy = 0;
									for (m = i ; m < i+p_C[convol]->sizeM ; m++) {
										for (n = j ; n < j+p_C[convol]->sizeN ; n++) {
											value += p_INP->x[m][n]*p_C[convol]->plane[k].e[xx][yy];
											counter++;
											yy++;
										}
										xx++;
										yy = 0;
									}
									GAMA = ETA/(MI+p_C[convol]->plane[k].hkk[0][i][j]);
									p_C[convol]->plane[k].w[0][i][j] += GAMA*value;//counter;
								}
								else {					//z predch. vrstvy
									for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
										xx = 0;
										yy = 0;
										for (m = i ; m < i+p_C[convol]->sizeM ; m++) {
											for (n = j ; n < j+p_C[convol]->sizeN ; n++) {
												value += p_S[sampl-1]->plane[h].x[m][n]*p_C[convol]->plane[k].e[xx][yy];
												counter++;
												yy++;
											}
											xx++;
											yy = 0;
										}
										if (p_C[convol]->plane[k].w[h][i][j] != 0.0) {
											GAMA = ETA/(MI+p_C[convol]->plane[k].hkk[h][i][j]);
											p_C[convol]->plane[k].w[h][i][j] += GAMA*value;//counter;
										}
									}
								}
							}
						}
					}
				
				convol++;
				break;
				case 'd':
				convol++;
				break;
				case 's':
				
					for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
						float value = 0.0;
						float error = 0.0;
						int counter = 0;
						for (i = 0 ; i < p_S[sampl]->sizeM ; i++) {
							for (j = 0 ; j < p_S[sampl]->sizeN ; j++) {
								error += p_S[sampl]->plane[k].e[i][j];//*(-1);
								counter++;
							}
						}
						p_S[sampl]->plane[k].bias += (ETA/(MI+p_S[sampl]->plane[k].biashkk))*error;//counter;
						//p_S[sampl]->plane[k].bias -= (ETA/MI)*error;//counter;
						counter = 0;
						for (i = 0 ; i < p_S[sampl]->receptiveM ; i++) {
							for (j = 0 ; j < p_S[sampl]->receptiveN ; j++) {
								for (m = i ; m < p_C[convol-1]->sizeM ; m = m+p_S[sampl]->receptiveM) {            // asi nie dobre
									for (n = j ; n < p_C[convol-1]->sizeN ; n = n+p_S[sampl]->receptiveN) {
										value += p_C[convol-1]->plane[k].x[m][n]*p_S[sampl]->plane[k].e[m/p_S[sampl]->receptiveM][n/p_S[sampl]->receptiveN];
										counter++;
									}
								}
							}
						}
						GAMA = ETA/(MI+p_S[sampl]->plane[k].hkk);
						p_S[sampl]->plane[k].w += GAMA*value;//counter;
					}
				
				sampl++;
				break;
				case 't':
				sampl++;
				break;
				case 'f':
				for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
					p_F[classic]->plane[k].bias += (ETA/(MI+p_F[classic]->plane[k].biashkk))*p_F[classic]->plane[k].e;//*(-1);
					//p_F[classic]->plane[k].bias -= (ETA/MI)*p_F[classic]->plane[k].e*(-1);
					int temp = 0;
					if (classic == 0) {
						for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
							for (i = 0 ; i < p_S[sampl-1]->sizeM ; i++) {
								for (j = 0 ; j < p_S[sampl-1]->sizeN ; j++) {
									GAMA = ETA/(MI+p_F[classic]->plane[k].hkk[temp]);
									p_F[classic]->plane[k].w[temp] += GAMA*p_F[classic]->plane[k].e*p_S[sampl-1]->plane[h].x[i][j];
									temp++;
								}
							}
						}
					}
					else {
						for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
							GAMA = ETA/(MI+p_F[classic]->plane[k].hkk[h]);
							p_F[classic]->plane[k].w[h] += GAMA*p_F[classic]->plane[k].e*p_F[classic-1]->plane[h].x;
						}
					}
				}
				classic++;
				break;
			}
		}
	}
	else {
		for (int classic = 0 ; classic < _CLASSICS ; classic++) {
			for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
				p_F[classic]->plane[k].bias += (ETA/(MI+p_F[classic]->plane[k].biashkk))*p_F[classic]->plane[k].e;//*(-1);
				//p_F[classic]->plane[k].bias -= (ETA/MI)*p_F[classic]->plane[k].e*(-1);
				int temp = 0;
				if (classic == 0) {
					for (h = 0 ; h < filter_size ; h++) {
						GAMA = ETA/(MI+p_F[classic]->plane[k].hkk[h]);
						p_F[classic]->plane[k].w[h] += GAMA*p_F[classic]->plane[k].e*filter_output[h];
						temp++;
					}
					
				}
				else {
					for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
						GAMA = ETA/(MI+p_F[classic]->plane[k].hkk[h]);
						p_F[classic]->plane[k].w[h] += GAMA*p_F[classic]->plane[k].e*p_F[classic-1]->plane[h].x;
					}
				}
			}
		}
	}

	if (rbf_patterns == 0) {
		for (k = 0 ; k < p_OUT->this_planes ; k++) {
			p_OUT->plane[k].bias += (ETA/(MI+p_OUT->plane[k].biashkk))*p_OUT->plane[k].e;//*(-1);
			//p_OUT->plane[k].bias -= (ETA/MI)*p_OUT->plane[k].e*(-1);
			for (h = 0 ; h < p_F[_CLASSICS-1]->this_planes ; h++) {
				GAMA = ETA/(MI+p_OUT->plane[k].hkk[h]);
				p_OUT->plane[k].w[h] += GAMA*p_OUT->plane[k].e*p_F[_CLASSICS-1]->plane[h].x;
			}
		}
	}
}

//******************************************************** UPRAVENE dokoncit
// funkcia na vypocet koef. z druhych derivacii
void WeakCNN::adjust_hessian(long EST_CYCLES)
{
	int k,i,j,m,n,h;
	int convol = 0;
	int sampl = 0;
	int classic = 0;
	float GAMA = 0;			//vysledny uciaci koef.
	
	if (filter_heredity == 0) {
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
			switch (info[a]) {
				case 'c':
				if (filter_heredity == 0) {
					for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
						
						if (convol == 0) {
							float biashess = 0.0;
							for (m = 0 ; m < p_C[convol]->sizeM ; m++) {
								for (n = 0 ; n < p_C[convol]->sizeN ; n++) {
									biashess += p_C[convol]->plane[k].e2[m][n];
								}
							}
							biashess *= p_INP->x[m][n]*p_INP->x[m][n];
							p_C[convol]->plane[k].biashee += biashess/EST_CYCLES;
						}
						else {
							float biashess = 0.0;
							for (m = 0 ; m < p_C[convol]->sizeM ; m++) {
								for (n = 0 ; n < p_C[convol]->sizeN ; n++) {
									biashess += p_C[convol]->plane[k].e2[m][n];
								}
							}
							for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
								biashess *= p_S[sampl-1]->plane[h].bias*p_S[sampl-1]->plane[h].bias;
							}
							p_C[convol]->plane[k].biashee += biashess/EST_CYCLES;

						}

						for (i = 0 ; i < p_C[convol]->receptiveM ; i++) {
							for (j = 0 ; j < p_C[convol]->receptiveN ; j++) {
								float hess = 0.0;
								int counter = 0.0;
								int xx = 0;
								int yy = 0;
								if (convol == 0) {
									xx = 0;
									yy = 0;
									for (m = i ; m < i+p_C[convol]->sizeM ; m++) {
										for (n = j ; n < j+p_C[convol]->sizeN ; n++) {
											hess += p_INP->x[m][n]*p_INP->x[m][n]*p_C[convol]->plane[k].e2[xx][yy];
											counter++;
											yy++;
										}
										xx++;
										yy = 0;
									}
									
									
									p_C[convol]->plane[k].hee[0][i][j] += hess/EST_CYCLES;
								}
								else {
									for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
										xx = 0;
										yy = 0;
										for (m = i ; m < i+p_C[convol]->sizeM ; m++) {
											for (n = j ; n < j+p_C[convol]->sizeN ; n++) {
												hess += p_S[sampl-1]->plane[h].x[m][n]*p_S[sampl-1]->plane[h].x[m][n]*p_C[convol]->plane[k].e2[xx][yy];
												counter++;
												yy++;
											}
											xx++;
											yy = 0;
										}
										if (p_C[convol]->plane[k].w[h][i][j] != 0.0) {
											
											p_C[convol]->plane[k].hee[h][i][j] += hess/EST_CYCLES;
										}
									}
								}
							}
						}
					}
				}
				convol++;
				break;
				case 'd':
				convol++;
				break;
				case 's':
				if (filter_heredity == 0) {
					for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
						float biashess = 0.0;
						int xx = 0;
						int yy = 0;
						
						for (m = 0 ; m < p_S[sampl]->sizeM ; m++) {
							for (n = 0 ; n < p_S[sampl]->sizeN ; n++) {
								//biashess += p_S[sampl]->plane[k].bias*p_S[sampl]->plane[k].bias*p_S[sampl]->plane[k].e2[m][n];
								biashess += p_S[sampl]->plane[k].e2[m][n];
							}	
						}
						biashess *= p_S[sampl]->plane[k].bias*p_S[sampl]->plane[k].bias;
						p_S[sampl]->plane[k].biashee += biashess/EST_CYCLES;
						
						float hess = 0.0;
						int counter = 0.0;
						for (i = 0 ; i < p_S[sampl]->receptiveM ; i++) {
							for (j = 0 ; j < p_S[sampl]->receptiveN ; j++) {
								int xx = 0;
								int yy = 0;
								for (m = 0 ; m < p_C[convol-1]->sizeM ; m = m+p_S[sampl]->receptiveM) {
									for (n = 0 ; n < p_C[convol-1]->sizeN ; n = n+p_S[sampl]->receptiveN) {
										hess += p_C[convol-1]->plane[k].x[m][n]*p_C[convol-1]->plane[k].x[m][n]*p_S[sampl]->plane[k].e2[m/p_S[sampl]->receptiveM][n/p_S[sampl]->receptiveN];
										counter++;
									}	
								}
							}
						}
						
						p_S[sampl]->plane[k].biashee += hess/EST_CYCLES;
						p_S[sampl]->plane[k].hee += hess/EST_CYCLES;
					}
				}
				sampl++;
				break;
				case 't':
				sampl++;
				break;
				case 'f':
				for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
					int temp = 0;
					if (classic == 0) {
						for (h = 0 ; h < p_S[sampl-1]->this_planes ; h++) {
							for (i = 0 ; i < p_S[sampl-1]->sizeM ; i++) {
								for (j = 0 ; j < p_S[sampl-1]->sizeN ; j++) {
									float hess = p_S[sampl-1]->plane[h].x[i][j]*p_S[sampl-1]->plane[h].x[i][j]*p_F[classic]->plane[k].e2;
									
									p_F[classic]->plane[k].hee[temp] += hess/EST_CYCLES; 
									temp++;
								}
							}
						}
						float biashess = p_S[sampl-1]->plane[h].bias*p_S[sampl-1]->plane[h].bias*p_F[classic]->plane[k].e2;
						p_F[classic]->plane[k].biashee += biashess/EST_CYCLES;
					}
					else {
						for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
							float hess = p_F[classic-1]->plane[h].x*p_F[classic-1]->plane[h].x*p_F[classic]->plane[k].e2;
							
							p_F[classic]->plane[k].hee[h] += hess/EST_CYCLES;
							
							float biashess = p_F[classic-1]->plane[h].bias*p_F[classic-1]->plane[h].bias*p_F[classic]->plane[k].e2;
							p_F[classic]->plane[k].biashee += biashess/EST_CYCLES;
						}
						//p_F[classic]->plane[k].biashee += p_F[classic]->plane[k].e2/EST_CYCLES;
					}
				}
				classic++;
				break;
			}
		}
	}
	else {
		for (int classic = 0 ; classic < _CLASSICS ; classic++) {	
			for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
				int temp = 0;
				if (classic == 0) {
					
					for (h = 0 ; h < filter_size ; h++) {
						float hess = filter_output[h]*filter_output[h]*p_F[classic]->plane[k].e2;
						
						p_F[classic]->plane[k].hee[temp] += hess/EST_CYCLES; 
						temp++;

						float biashess = filter_output[h]*filter_output[h]*p_F[classic]->plane[k].e2;
						p_F[classic]->plane[k].biashee += biashess/EST_CYCLES;
					}
					//float biashess = filter_output[h]*filter_output[h]*p_F[classic]->plane[k].e2;
					//p_F[classic]->plane[k].biashee += biashess/EST_CYCLES;

				}
				else {
					for (h = 0 ; h < p_F[classic-1]->this_planes ; h++) {
						float hess = p_F[classic-1]->plane[h].x*p_F[classic-1]->plane[h].x*p_F[classic]->plane[k].e2;
						
						p_F[classic]->plane[k].hee[h] += hess/EST_CYCLES;
						
						float biashess = p_F[classic-1]->plane[h].bias*p_F[classic-1]->plane[h].bias*p_F[classic]->plane[k].e2;
						p_F[classic]->plane[k].biashee += biashess/EST_CYCLES;
					}
					//p_F[classic]->plane[k].biashee += p_F[classic]->plane[k].e2/EST_CYCLES;
				}
			}
		}
	}
	if (rbf_patterns == 0) {
		for (k = 0 ; k < p_OUT->this_planes ; k++) {
			for (h = 0 ; h < p_F[_CLASSICS-1]->this_planes ; h++) {
				float hess = p_F[_CLASSICS-1]->plane[h].x*p_F[_CLASSICS-1]->plane[h].x*p_OUT->plane[k].e2;
				
				p_OUT->plane[k].hee[h] += hess/EST_CYCLES;
				
				float biashess = p_F[_CLASSICS-1]->plane[h].bias*p_F[_CLASSICS-1]->plane[h].bias*p_OUT->plane[k].e2;
				p_OUT->plane[k].hee[h] += biashess/EST_CYCLES;
			}
			//p_F[classic]->plane[k].biashee += p_F[classic]->plane[k].e2/EST_CYCLES;
		}
	}
}

//******************************************************** UPRAVENE dokoncit
//funkcia na skopirovanie koef. po skonceni estimacie 

void WeakCNN::copy_hessians(void)
{
	int k,i,j,m,n,h;
	int convol = 0;
	int sampl = 0;
	int classic = 0;
	for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
		switch (info[a]) {
			case 'c':
			for (k = 0 ; k < p_C[convol]->this_planes ; k++) {
				p_C[convol]->plane[k].biashkk = p_C[convol]->plane[k].biashee;
				for (i = 0 ; i < p_C[convol]->receptiveM ; i++) {
					for (j = 0 ; j < p_C[convol]->receptiveN ; j++) {
						int max;
						if (convol == 0)
							max = 1;	//bereme zo vstupu
						else
							max = p_S[sampl-1]->this_planes;	//z predch. vrstvy
						for (h = 0 ; h < max ; h++) {
							if (p_C[convol]->plane[k].w[h][i][j] != 0.0) {
								p_C[convol]->plane[k].hkk[h][i][j] = p_C[convol]->plane[k].hee[h][i][j];
							}
						}
					}
				}
			}
			convol++;
			break;
			case 'd':
			convol++;
			break;
			case 'e':
			convol++;
			break;
			case 's':
			for (k = 0 ; k < p_S[sampl]->this_planes ; k++) {
				p_S[sampl]->plane[k].biashkk = p_S[sampl]->plane[k].biashee;
				p_S[sampl]->plane[k].hkk = p_S[sampl]->plane[k].hee;
			}
			sampl++;
			break;
			case 't':
			sampl++;
			break;
			case 'u':
			sampl++;
			break;
			case 'f':
			for (k = 0 ; k < p_F[classic]->this_planes ; k++) {
				for (h = 0 ; h < p_F[classic]->connections ; h++) {
					p_F[classic]->plane[k].hkk[h] = p_F[classic]->plane[k].hee[h];
				}
				p_F[classic]->plane[k].biashkk = p_F[classic]->plane[k].biashee;
			}
			classic++;
			break;
		}
	}
	
	if (rbf_patterns == 0) {
		for (k = 0 ; k < p_OUT->this_planes ; k++) {
			for (h = 0 ; h < p_OUT->connections ; h++) {
				p_OUT->plane[k].hkk[h] = p_OUT->plane[k].hee[h];
			}
			p_OUT->plane[k].biashkk = p_OUT->plane[k].biashee;
		}
	}

}

void WeakCNN::assign_input(read_file *p_vstup, long what)
{
	if (filter_heredity == 0) {
		int x_scale = (p_INP->sizeM - p_vstup->x_size) / 2;
		int y_scale = (p_INP->sizeN - p_vstup->y_size) / 2;
		for (int i = 0 ; i < p_vstup->x_size ; i++) {
			for (int j = 0 ; j < p_vstup->y_size  ; j++) {
				//normalizacia z (0,255) na (-0.1,1.175)
				//pre tanh
				if (act_function == 0) {
					p_INP->x[i+x_scale][j+y_scale] = ((float)p_vstup->pattern[what].pixel[i][j] / 200.0) - 0.1;
				}
				//normalizacia z (0,255) na (0,1)
				//pre sigmoidu
				else {
					p_INP->x[i+x_scale][j+y_scale] = ((float)p_vstup->pattern[what].pixel[i][j] / 255.0);
				}			
			}
		}
	}
	else {
		for (int i = 0 ; i < filter_size ; i++) {
			filter_output[i] = p_vstup->pattern[what].convolution_filter[dataset][i];
		}
			
	}
}

void WeakCNN::get_output(int radius)
{
	if (rbf_patterns == 0) {
		int temp = 0;
		float max = p_OUT->plane[0].x;
		
		for (int i = 1 ; i < p_OUT->this_planes ; i++) {
			if ( max < p_OUT->plane[i].x) {
				temp = i;
				max = p_OUT->plane[i].x;
			}
		}
		temporal_class = p_OUT->plane[temp].membership;
		temporal_activation = max;
		
		//tanh
		//-1 = confidence 0
		//1 = confidence 1
		if (act_function == 0) {
			if (temporal_activation > 1.0) {
				temporal_confidence = 1.0;
			}
			else if (temporal_activation < -1.0) {
				temporal_confidence = 0.0;
			}
			else {
				temporal_confidence = (temporal_activation + (float)1.0) / (float)2.0;
			}
			
		}
		//sigmoida 
		//0 - confidence 0
		//1 - confidence 1
		else {
			temporal_confidence = temporal_activation;
			
		}
	}
	else {
		int temp = 0;
		
		//_CLASSIC-1 je index predposlednej vrstvy 
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			float value = 0.0;
			for (int j = 0 ; j < p_F[_CLASSICS-1]->this_planes ; j++) {
				value += (p_F[_CLASSICS-1]->plane[j].x-p_OUT->plane[i].w[j])*(p_F[_CLASSICS-1]->plane[j].x-p_OUT->plane[i].w[j]);
				//value += (p_OUT->plane[i].w[j]-p_F[_CLASSICS-1]->plane[j].x)*(p_OUT->plane[i].w[j]-p_F[_CLASSICS-1]->plane[j].x);
			}
			p_OUT->plane[i].x = value;
		}

		float min = p_OUT->plane[0].x;
	
		for (int i = 1 ; i < p_OUT->this_planes ; i++) {
			if ( min > p_OUT->plane[i].x) {
				temp = i;
				min = p_OUT->plane[i].x;
			}
		}
		temporal_class = p_OUT->plane[temp].membership;
		temporal_activation = min;
		temporal_confidence = exp(-temporal_activation / radius);
	}
}

bool WeakCNN::output(int neuron, int &membership, float &activation)
{
	if (neuron < p_OUT->this_planes && neuron >= 0) {
		membership = p_OUT->plane[neuron].membership;	
		activation = p_OUT->plane[neuron].x;
		return true;
	}
	else {
		return false;
	}
}

double WeakCNN::get_binary_output(int item)
{
	if (p_F[_CLASSICS-1]->plane[item].x > 0) {
		return 1.0;
	}
	else {
		return -1.0;
	}
}

double WeakCNN::get_bit_output(int item)
{
	return p_F[_CLASSICS-1]->plane[item].x;
}


void WeakCNN::get_confidence(float radius)
{
	if (rbf_patterns == 0) {
		//tanh
		//-1 = confidence 0
		//1 = confidence 1
		if (act_function == 0) {
			for (int i = 0 ; i < p_OUT->this_planes ; i++) {
				if (p_OUT->plane[i].x > 1.0) {
					p_OUT->plane[i].conf = 1.0;
				}
				else if (p_OUT->plane[i].x < -1.0) {
					p_OUT->plane[i].conf = 0.0;
				}
				else {
					p_OUT->plane[i].conf = (p_OUT->plane[i].x + (float)1.0) / (float)2.0;
				}
			}
		}
		//sigmoida 
		//0 - confidence 0
		//1 - confidence 1
		else {
			for (int i = 0 ; i < p_OUT->this_planes ; i++) {
				p_OUT->plane[i].conf = p_OUT->plane[i].x;
			}
		}
	}
	else {
		//nezavisle od aktivacnej funkcie
		//SOFTMAX nie je velmi vhodne riesenie
		/*
		double sum = 0.0;
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			sum += exp(p_OUT->plane[i].x);
		}
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			p_OUT->plane[i].conf = (double)1.0 - (exp(p_OUT->plane[i].x) / sum);
		}
		*/

		//novozavedeny RADIUS
		double sum = 0.0;
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			p_OUT->plane[i].conf = exp(-p_OUT->plane[i].x / radius);
			sum += p_OUT->plane[i].conf;
		}
		/*
		for (int i = 0 ; i < p_OUT->this_planes ; i++) {
			p_OUT->plane[i].conf /= sum;
		}
		*/
	}
}

void WeakCNN::apply_masks(masks* temp)
{
	//pre vsetky dvojicekonvolucno vzorkovacich vrstiev
	for (int i = 0 ; i < _CONVOLUTIONS ; i++) {
		int mask_count = 0;
		//aplikacia masiek zaciname maskou 0 a pokracujeme dokym nepokryjeme plany aktualnej vrstvy
		
		for (int k = 0 ; k < p_C[i]->this_planes ; k++) {		
			
			for (int h = 0 ; h < p_C[i]->prev_planes ; h++) {
				int h_rand = (int)((rand()%(int)(temp->number_of_masks)));
				for (int m = 0 ; m < p_C[i]->receptiveM ; m++) {
					for (int n = 0 ; n < p_C[i]->receptiveN ; n++) {		
						/*
						//zasebou dokola
						if (i == 0) {
							p_C[i]->plane[k].w[h][m][n] = (float)temp->pattern[mask_count].pixel[m][n] / (float)p_C[i]->prev_planes;	
						}
						//alebo nahodne
						else {
							p_C[i]->plane[k].w[h][m][n] = (float)temp->pattern[h_rand].pixel[m][n] / (float)p_C[i]->prev_planes;
						}
						*/
						p_C[i]->plane[k].w[h][m][n] = (float)temp->pattern[h_rand].pixel[m][n] / (float)(p_C[i]->prev_planes*2.0);
					}
				}
				mask_count++;
				if (mask_count == temp->number_of_masks) {
					mask_count = 0;
				}
			}
			
		}

		for (int k = 0 ; k < p_S[i]->this_planes ; k++) {
			p_S[i]->plane[k].w = 1.0;
		}
		
	}

	//po aplikacii mask sa nastavi pole info tak, aby sa :
	//signal siril dopredu v celej sieti
	//chybovy signal nesiril v cati filtra
	//vahy sa neadaptovali v casti filtra
	for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
		if (info[a] == 'c') 
			info[a] = 'd';
		if (info[a] == 's')
			info[a] = 't';
	}
}

void WeakCNN::apply_heredity(bool flag)
{
	if (flag == true) {
		//po aplikacii heredity sa nastavi pole info tak, aby sa :
		//signal siril dopredu od vystupu filtra
		//chybovy signal nesiril v cati filtra
		//vahy sa neadaptovali v casti filtra
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
			if (info[a] == 'c') 
				info[a] = 'e';
			if (info[a] == 's')
				info[a] = 'u';
		}
		filter_heredity = 1;
	}
	else {
		//nastavi sa do povodneho stavu
		for (int a = 0 ; a < _CONVOLUTIONS+_SAMPLINGS+_CLASSICS ; a++) {
			if (info[a] == 'e') 
				info[a] = 'c';
			if (info[a] == 'u')
				info[a] = 's';
		}
		filter_heredity = 0;

	}
}

void WeakCNN::initialize_rbf(rbfpatterns *rbfPAT)
{
	int i,j,k,h;
	//_CLASSIC-1 je index predposlednej vrstvy
	for (k = 0 ; k < p_OUT->this_planes ; k++) {
		//privedenie vzorky na vstup RBF, co je predposledna vrstva
		int temp = 0;
		for (i = 0 ; i < rbfPAT->size_x ; i++) {
			for (j = 0 ; j < rbfPAT->size_y ; j++) {
				p_F[_CLASSICS-1]->plane[temp].x = (float)rbfPAT->pattern[p_OUT->plane[k].membership].pixel[i][j];
				temp++;
			}
		}
		for (h = 0 ; h < p_F[_CLASSICS-1]->this_planes ; h++) {
			if (act_function == 0) {
				if (p_F[_CLASSICS-1]->plane[h].x == 1)
					p_OUT->plane[k].w[h] = 1.0;//0.88;
				else if (p_F[_CLASSICS-1]->plane[h].x == -1)
					p_OUT->plane[k].w[h] = -1.0;//-0.9;
				else
					p_OUT->plane[k].w[h] = 0.0;//-0.9;
			}
			else {
				if (p_F[_CLASSICS-1]->plane[h].x == 1)
					p_OUT->plane[k].w[h] = 1.0;//0.88;
				else if (p_F[_CLASSICS-1]->plane[h].x == 0)
					p_OUT->plane[k].w[h] = 0;//-0.9;
				else
					p_OUT->plane[k].w[h] = 0.5;//-0.9;
			}
		}
	}
}

