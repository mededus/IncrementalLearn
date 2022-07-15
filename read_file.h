//********************************************************
//read_file.h
//
//Trieda pre uciace(trenovacie) vzorky

//********************************************************
#pragma once
#include <math.h>

class read_file				//trieda uciacich prikladov
{
public:
	FILE *f, *g, *h;				//kompatibilna s MNIST formatom
	long magic_number;				//pomocna premenna - pocetnost tried
	long patterns;					//pocet prikladov v subore
	int x_size;						//rozmer X
	int y_size;						//rozmer Y
	int *classes;					//informacia o pocetnosti prvkov v triedach
	struct input {					//struktura prikladov
		unsigned char **pixel;		//obrazove data 
		int membership;				//trieda prikladu 
		int length;					//dlzka (ak existuje)
		double weight;				//vaha prikladu v zmysle Boostingu
		double weightN;				//distribucia - normalizovana vaha v zmysle SSwR (aby bol obvod stale =1)
		double weightR;				//pomocne vahy pre SSwoR ruletu
		double *misslabel;			//vahy prikladov v zmysle pseudoloss Boostingu
		double *misslabelN_loc;		//distribucia - normalizovane pre lokalnu hypotezu
		double *misslabelN_glo;		//distribucia - normalizovane pre zlozenu hypotezu
		short repl_flag;			//priznak vybrartia prikladu ruletou
		float **convolution_filter;	//aktivita neuronov po prechode konvolucnym filtrom
		int **Htfield;				//pomocna premenna - vysledok binarnej hypotezy 
		double **Htfieldd;
		int Ht;						//pomocna premenna - udrzuje vysledok hypotezy
		double *Hconf;				//confidence pre aktualnu hypotezu
	} *pattern;						//pole prikladov
	int ht_size_x;					//pomocne premenne
	int ht_size_y;
	int cfilter_size_x;
	int cfilter_size_y;

	read_file(void);					//konstruktor
	~read_file(void);					//destruktor
	bool open_file(char *, char *, char *);	
										//nacitanie suborov "*.images", "*.labels" a "*.lenghts"
	void read_header(void);				//macitanie hlavicky
	void read_patterns(void);			//nacitanie dat
	void reset_weights(void);			//vynulovanie distribucnych vah
	void normalize_weights(void);		//normalizacia vah 
	void set_weight(long, double);		//rucne nastavenie vahy
	void setup_convolution(int, int);	//mapuje vystupy z "extraktora priznakov"
	void setup_Htfield(int, int);		//pomocna funkcia
	void setup_misslabels(int, int *);	//vytvorenie distribucii B a C
	void apply_misslabels(void);		//nastavenie distribucii B a C
	void global_misslabels(void);		//nastavenie globalnych distribucii Dc(t)
	void local_misslabels(int, int *);	//nastavenie lokalnych distribucii D(t)
};
