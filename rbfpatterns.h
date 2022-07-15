#pragma once

class rbfpatterns				//trieda distribucnych kodov
{
public:
	FILE *f;					//smernik na subor distribucnych kodov
	int size_x;					//rozmer x kodu
	int size_y;					//rozmer y kodu
	int number_of_pat;			//pocet distribucnych kodov = pocet tried

	struct patterns {			//struktura distribucneho kodu
		int **pixel;			//body dist. kodu
		int code;				//trieda - prislusnost
	} *pattern;					//smernik
	
	rbfpatterns(void);			//konstruktor
	~rbfpatterns(void);			//destruktor
	void print_pattern(int);	//vypis patternu do konzoly
	bool open_file(char *);		//otvorenie suboru "*.pattern"
	void read_patterns(short act_function);	
								//nasitanie suborov do udajovych struktur
};