#pragma once

class masks					//trieda statickych masiek
{
public:

	FILE *f;				//smernik na subor
	int size_x;				//rozmer X masky
	int size_y;				//rozmer Y masky
	int number_of_masks;
		
	struct patterns {		//mnozina masiek
		int **pixel;		//body masky
		int number;			//poradove cislo
	} *pattern;				//smernik na mnozinu

	masks(void);			//konstruktor
	~masks(void);			//destruktor
	bool open_file(char *);	//otvorenie subor "masks"
	void read_patterns(void);	
							//nasitanie obsahu suboru "masks"
};
