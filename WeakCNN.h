#pragma once
#include "classic_layer.h"
#include "convol_layer.h"
#include "input_layer.h"
#include "sampl_layer.h"
#include "read_file.h"
#include "masks.h"
#include "rbfpatterns.h"
#include <math.h>

class WeakCNN
{
public:
	WeakCNN(int, int, int, short, short);	//konstruktor
	~WeakCNN(void);							//destruktor
	
	//premenne
	int _CONVOLUTIONS;						//pocet konvolucnych vrstiev			
	int _SAMPLINGS;							//pocet vzorkovacich vrstiev
	int _CLASSICS;							//pocet klasickych skrytych vrstiev
	
	float ETA;								//uciace koef.
	float MI;
	
	char *info;								//nesie info o type vrstiev

	input_layer		*p_INP; 				//smerniky na vstup a vystup
	classic_layer	*p_OUT;				
	convol_layer	**p_C;					//smerniky na vnutorne vrstvy
	sampl_layer		**p_S;		
	classic_layer	**p_F;

	read_file		*p_vstup;				//smernik na subor uciacej(trenovacej) vzorky

											//parametre tykajuce sa boostingu
	double epsilon;							//chyba klasifikatora
	double composite_epsilon;
	double beta;							//normalizovana chyba
	double composite_beta;
	double weight;							//vaha klasifikatora
	double composite_weight;
	double weightN;							//normalizovana vaha vramci aktualneho suboru
	double composite_weightN;
	double dwvweight;						//vaha pre DVW algoritmus
	//bool status;	//potom vymazat
	int dataset;							//informacia o tom na ktorom datasete bol uceny

	int filter_size;						//velkost vystupu (rozmer) z filtra
	float *filter_output;					//vystup z filtra (prva vrstva, kt. nie je organizovana do planov)
	int filter_heredity;					//priznak filtra

	double wmvweight;						//vaha pre WMV algoritmus

	int *classes;							//pouzite triedy 
	int number_of_classes;					//pocet tried
	int misclassification;					//pocet zle klasifikovanych vzozov na trenovacke
	int composite_misclassification;
	short act_function;						//aktivacna funkcia
	short rbf_patterns;						//rbf patterny
	
	int temporal_class;						//docasna vystupna trieda vitaza
	float temporal_activation;				//docasna vystupne aktivacia vitaza
	float temporal_confidence;				//docasna confidence vitaza
	
	// metody
	int initialize_net(char* name, bool initialize);	
										//inicializacia
	int save_net(char* name);			//ulozenie siete
	float squash_f(float value);		//aktivacna funkcia
	float squash_df(float value);		//1. derivacia aktivacnej funkcie
	float squash_ddf(float value);		//2. derivacia aktivacnej funkcie
	void feed_forward(void);			//presirenie signalu sietou
	void back_propagation_e(int membership, rbfpatterns *rbfPAT, float &output_error, float &bpe_error);
										//spatne sirenie chyby sietou
	void back_propagation_e2(int membership);
										//spatne sirenie derivacie chyby sietou 
	void adjust_weights(float ETA, float MI);
										//adaptacia vah
	void adjust_hessian(long EST_CYCLES);
										//adaptacia lokalnych parametrov ucenia
	void copy_hessians(void);			//pomocna funckia pre ukladanie lokalnych parametrov
	void assign_input(read_file *p_vstup, long what);
										//priradenie vstupu
	void get_output(int);				//zistenie vystupu (vitaza) zo siete
	bool output(int neuron, int &membership, float &activation);
										//zistenie vystupu pozadovaneho neuronu
	double get_binary_output(int);		//zistenie vystupu z distribucnej vrstvy
	double get_bit_output(int);			//pomocna funckia 
	void get_confidence(float);			//vypocet doveryhosnosti na vystupe
	void apply_masks(masks *temp);		//aplikacia statickych konvolucnych filtrov 
	void apply_heredity(bool);			//aplikacia zdielaneho extraktora priznakov
	void initialize_rbf(rbfpatterns *rbfPAT);
										//inicializacia distribucnych kodov
	int reset_net(void);				//reinicializacia siete
};
