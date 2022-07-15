#include "StdAfx.h"
#include "masks.h"

masks::masks(void)
{
	number_of_masks = 0;
	size_x = 0;
	size_y = 0;
	f = NULL;
	pattern = NULL;
}

masks::~masks(void)
{
	delete[] pattern;
}

bool masks::open_file(char *name)
{
	if( !(f=fopen(name,"rt"))) 
		return(false);
	else	
		return(true);
}

void masks::read_patterns(void)
{
	char temp[100];

	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &number_of_masks);
	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &size_x);
	fscanf(f, "%s", &temp);
	fscanf(f, "%d", &size_y);
	fscanf(f, "%s", &temp);
	
	pattern = new patterns[number_of_masks];
	for (int a = 0 ; a < number_of_masks ; a++) {
		pattern[a].pixel = new int*[size_x];
		for (int b = 0 ; b < size_x ; b++) {
			pattern[a].pixel[b] = new int [size_y];
		}
	}
	for (int k = 0 ; k < number_of_masks ; k++) {
		fscanf(f, "%d", &pattern[k].number);
		for (int i = 0 ; i < size_x ; i++) {
			for (int j = 0 ; j < size_y ; j++) {
				fscanf(f, "%d", &pattern[k].pixel[i][j]);
				
			}
		}
	}
	/* print
	for (int k = 0 ; k < number_of_masks ; k++) {
		printf("\n\n");	
		for (int i = 0 ; i < size_x ; i++) {
			printf("\n");
			for (int j = 0 ; j < size_y ; j++) {
					printf("%d ", pattern[k].pixel[i][j]);
			}
		}
	}
	*/
	fclose(f);		
}
