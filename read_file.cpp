//********************************************************
//read_file.cpp
//
//Metody triedy pre uciace(trenovacie) vzorky

//********************************************************
#include "StdAfx.h"
#include "read_file.h"

read_file::read_file(void)
{
	magic_number = 0;
	patterns = 0;
	x_size = 0;
	y_size = 0;	
	f = NULL;
	g = NULL;
	h = NULL;
	pattern = NULL;
	ht_size_x = 0;					
	ht_size_y = 0;
	cfilter_size_x = 0;
	cfilter_size_y = 0;
}

read_file::~read_file(void)
{
	if (pattern != NULL) {
		for ( int a = 0 ; a < patterns ; a++) {
			for ( int d = 0 ; d < x_size ; d++) {
				delete[] pattern[a].pixel[d];

			}
			delete[] pattern[a].pixel;
			delete[] pattern[a].misslabel;
			delete[] pattern[a].misslabelN_loc;
			delete[] pattern[a].misslabelN_glo;

			if (cfilter_size_x != 0) {
				for (int d = 0 ; d < cfilter_size_x ; d++) {
					delete[] pattern[a].convolution_filter[d];
				}
				delete[] pattern[a].convolution_filter;
			}
			
			if (ht_size_x != 0) {
				for (int d = 0 ; d < ht_size_x ; d++) {
					delete[] pattern[a].Htfield[d];
					delete[] pattern[a].Htfieldd[d];
				}
				delete[] pattern[a].Htfield;
				delete[] pattern[a].Htfieldd;
			}

			if (pattern[a].Hconf != NULL) {
				delete[] pattern[a].Hconf;
			}

		}
		delete[] classes;
		delete[] pattern;
	}
}

bool read_file::open_file(char *train_name, char *membership_name, char *lenghts_name)
{
	//pripadne sa otvori subor *.lengths
	h=fopen(lenghts_name,"rb");
	
	if( !(f=fopen(train_name,"rb")) || !(g=fopen(membership_name,"rb"))) 
		return(false);
	else	
		return(true);
}

void read_file::read_header(void)
{
	long temp[4];
	fseek(f,0,0); //magic_number na prvych 4 bajtoch
	for ( int i = 0 ; i < 4 ; i++) temp[i] = fgetc(f);
	magic_number = temp[0]*pow(16,6)+temp[1]*pow(16,4)+temp[2]*pow(16,2)+temp[3];
	//fseek(f,4,0); //pocet vzoriek na druhych 4 bajtoch
	for ( int i = 0 ; i < 4 ; i++) temp[i] = fgetc(f);
	patterns = temp[0]*pow(16,6)+temp[1]*pow(16,4)+temp[2]*pow(16,2)+temp[3];
	//fseek(f,8,0); //rozmer vzorky na tretich 4 bajtoch
	for ( int i = 0 ; i < 4 ; i++) temp[i] = fgetc(f);
	x_size = temp[0]*pow(16,6)+temp[1]*pow(16,4)+temp[2]*pow(16,2)+temp[3];
	//fseek(f,12,0); //rozmer vzorky na stvrtych 4 bajtoch
	for ( int i = 0 ; i < 4 ; i++) temp[i] = fgetc(f);
	y_size = temp[0]*pow(16,6)+temp[1]*pow(16,4)+temp[2]*pow(16,2)+temp[3];

	classes = new int[255];	//max pocet tried FF(hexa)
	for (int i = 0 ; i < 255 ; i++) classes[i] = 0;
}

void read_file::read_patterns(void)
{
	pattern = new input[patterns];
	for (int i = 0 ; i < patterns ; i++) {
		pattern[i].pixel = new unsigned char*[x_size];
		for (int j = 0 ; j < x_size ; j++) {
			pattern[i].pixel[j] = new unsigned char[y_size];
		}
		pattern[i].misslabel = new double[255];
		pattern[i].misslabelN_loc = new double[255];
		pattern[i].misslabelN_glo = new double[255];
		pattern[i].Hconf = NULL;
	}

	fseek(f,16,0);	//nastavenie na prvu obrazovu info v f
	
	for (int k = 0 ; k < patterns ; k++) {
		for (int i = 0 ; i < y_size ; i++) {
			for (int j = 0 ; j < x_size ; j++) {
				pattern[k].pixel[j][i] = fgetc(f);
			}
		}
		//pattern[k].weight = pattern[k].weightN = (double)1.0 / (double)patterns;	// nastavenie pociatocnych vah

	}

	fseek(g,8,0);	//nastavenie na prvu obrazovu info v g
	for (int k = 0 ; k < patterns ; k++) {
		pattern[k].membership = fgetc(g);	
	}
	
	if (h != NULL) {
		fseek(h,8,0);	//nastavenie na prvu obrazovu info v h
		for (int k = 0 ; k < patterns ; k++) {
			pattern[k].length = fgetc(h);	
		}
	}

	//zistime ake triedy sa nachadzaju v subore
	for (int k = 0 ; k < patterns ; k++) {
		classes[pattern[k].membership]++;
	}
	//zistime ich pocetnost
	if (magic_number == 0) {
		for (int k = 0 ; k < 255 ; k++) {
			if (classes[k] > 0) {
				magic_number++;
			}
		}
	}

	fclose(f);
	fclose(g);
}

void read_file::reset_weights(void)
{
	
	//klasicky pristup 1 / pocet prikladov
	for (int k = 0 ; k < patterns ; k++) {
		pattern[k].weight = pattern[k].weightN = (double)1.0 / (double)patterns;	// nastavenie pociatocnych vah

	}
	
}

void read_file::normalize_weights(void)
{
	
	double sum = 0.0;
	for (int k = 0 ; k < patterns ; k++) {
		sum += pattern[k].weight;
	}

		for (int k = 0 ; k < patterns ; k++) {
		pattern[k].weightN = (double)pattern[k].weight / sum;
	}
	
	sum = 0.0;
	for (int k = 0 ; k < patterns ; k++) {
		sum += pattern[k].weightN;
	}
	
}

void read_file::set_weight(long index, double value)
{
	pattern[index].weight = value;
}

void read_file::setup_convolution(int count, int size)
{
	if (pattern != NULL) {
		for (int i = 0 ; i < patterns ; i++) {
			pattern[i].convolution_filter = new float*[count];
			for (int j = 0 ; j < count ; j++) {	
				pattern[i].convolution_filter[j] = new float[size];
			}
		}

		cfilter_size_x = count;
		cfilter_size_y = size;
	}
}

void read_file::setup_Htfield(int size_x, int size_y)
{
	if (pattern != NULL) {
		for (int k = 0 ; k < patterns ; k++) {
			pattern[k].Htfield = new int*[size_x];
			pattern[k].Htfieldd = new double*[size_x];
			for (int i = 0 ; i < size_x ; i++) {
				pattern[k].Htfield[i] = new int[size_y];
				pattern[k].Htfieldd[i] = new double[size_y];
			}
		}
		ht_size_x = size_x;					
		ht_size_y = size_y;
	}
}

void read_file::setup_misslabels(int count, int *list)
{
	if (pattern != NULL) {
		for (int k = 0 ; k < patterns ; k++) {
			for (int c = 0 ; c < 255 ; c++) {
				//globalne
				if (list[c] > 0) {
					if (pattern[k].membership == c) {
						pattern[k].misslabel[c] = 0.0;
						pattern[k].misslabelN_glo[c] = 0.0;
					}
					else {
						pattern[k].misslabel[c] = pattern[k].misslabelN_glo[c] = (double)1.0 / (double)(patterns * (count - 1));
					}
				}
				else {
					pattern[k].misslabel[c] = 0.0;
					pattern[k].misslabelN_glo[c] = 0.0;
				}
				//lokalne
				if (classes[c] > 0) {
					if (pattern[k].membership == c) {
						pattern[k].misslabelN_loc[c] = 0.0;
					}
					else {
						pattern[k].misslabelN_loc[c] = (double)1.0 / (double)(patterns * (magic_number - 1));
					}
				}
				else {
					pattern[k].misslabelN_loc[c] = 0.0;
				}
			}
		}
	}
}


void read_file::apply_misslabels(void)
{
	//vaha
	for (int k = 0 ; k < patterns ; k++) {
		double misslabels_sum = 0.0;
		for (int c = 0 ; c < 255 ; c++) {
			misslabels_sum += pattern[k].misslabel[c];
		}
		pattern[k].weight = misslabels_sum;
	}
	//distribucia
	double total_misslabels_sum = 0.0;
	for (int k = 0 ; k < patterns ; k++) {
		double misslabels_sum = 0.0;
		for (int c = 0 ; c < 255 ; c++) {
			misslabels_sum += pattern[k].misslabelN_glo[c];
		}
		pattern[k].weightN = misslabels_sum;
		total_misslabels_sum += misslabels_sum;
	}
	
}



void read_file::global_misslabels(void)
{
	double sum = 0.0;
	for (int k = 0 ; k < patterns ; k++) {
		for (int c = 0 ; c < 255 ; c++) {
			sum += pattern[k].misslabel[c];
		}
	}
	for (int k = 0 ; k < patterns ; k++) {
		for (int c = 0 ; c < 255 ; c++) {
			pattern[k].misslabelN_glo[c] = (double)pattern[k].misslabel[c] / sum;
		}
	}
	
}

void read_file::local_misslabels(int count, int *list)
{
	double summ = 0.0;
	for (int k = 0 ; k < patterns ; k++) {
		//spocitaju sa diely (triedy) ktore dany klasifikator nepozna
		double val = 0.0;
		for (int c = 0 ; c < 255 ; c++) {
			if (list[c] > 0 && classes[c] == 0) {
				val += pattern[k].misslabelN_glo[c];
			}
		}
		//ziskana hodnota sa deli poctom misslabelov na datasete
		val /= (double)(magic_number - 1);
		
		
		//vypocet lokalnych hodnot
		for (int c = 0 ; c < 255 ; c++) {
			//if (classes[c] > 0) {
			if (pattern[k].misslabelN_loc[c] > 0.0) { 
				pattern[k].misslabelN_loc[c] = pattern[k].misslabelN_glo[c] + val; 
			}
			summ += pattern[k].misslabelN_loc[c];
		}
	}
	
}
