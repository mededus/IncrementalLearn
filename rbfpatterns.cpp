#include "StdAfx.h"
#include "rbfpatterns.h"

rbfpatterns::rbfpatterns(void)
{
	number_of_pat = 0;
	size_x = 0;
	size_y = 0;
	f = NULL;
	pattern = NULL;
}

rbfpatterns::~rbfpatterns(void)
{
	if (pattern != NULL) {
		for ( int a = 0 ; a < number_of_pat ; a++) {
			for ( int d = 0 ; d < size_x ; d++) {
				delete[] pattern[a].pixel[d];

			}
			delete[] pattern[a].pixel;
			
		}
		delete[] pattern;
	}
}

void rbfpatterns::print_pattern(int number)
{
	printf("\n\n");
	for (int i = 0 ; i < size_x ; i++) {
		printf("\n");
		for (int j = 0 ; j < size_y ; j++) {
			if (pattern[number].pixel[i][j] > 0)
				printf("%d ", 1);
			else
				printf("%d ", 0);
			//printf("%d ", pattern[number].pixel[i][j]);
		}
	}
}
bool rbfpatterns::open_file(char *name)
{
	if( !(f=fopen(name,"rt"))) 
		return(false);
	else	
		return(true);
}

void rbfpatterns::read_patterns(short act_function)
{
	char temp[100];
	int pat;

	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &number_of_pat);
	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &size_x);
	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &size_y);
	fscanf(f, "%s", &temp);
	
	pattern = new patterns[number_of_pat];
	for (int a = 0 ; a < number_of_pat ; a++) {
		pattern[a].pixel = new int*[size_x];
		for (int b = 0 ; b < size_x ; b++) {
			pattern[a].pixel[b] = new int [size_y];
		}
	}
	for (int k = 0 ; k < number_of_pat ; k++) {
		fscanf(f, "%d", &pattern[k].code);
		for (int i = 0 ; i < size_x ; i++) {
			for (int j = 0 ; j < size_y ; j++) {
				fscanf(f, "%d", &pat);
				if ( pat == 0 ) {
					if (act_function == 0)
						pattern[k].pixel[i][j] = -1;
					else
						pattern[k].pixel[i][j] = 0;
				}
				else if ( pat == 1) {
					pattern[k].pixel[i][j] = 1;
				}
			}
		}
	}

	
	fclose(f);		
}