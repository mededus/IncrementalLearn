// Incremental.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "atlfile.h"
#include "WeakCNN.h"
#include "randomc.h"
#include <conio.h>
#include <time.h>

//*******************************
//definicia globalnych premennych

HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);	//ukazovatel na konzolu
CONSOLE_SCREEN_BUFFER_INFO csbiInfo;				//info o konzole
COORD coordScreen = { 0, 0 };						//info o polohe kurzora

TRanrotBGenerator generator((unsigned)time(NULL));	//inicializacia generatora rulety

char working_directory[255];						//pracovny adresar

//parametre algoritmu
int alg_iterations = 20;		  					//kolko maximalne klasifikatorov pre dataset
int alg_datasets = 1;								//kolko datasetov
int alg_samples = 0;    							//kolko prikladov na trenovaciu mnozinu
float alg_part_samples = 0.0;						//aka cast mnoziny sa pouzije ako trenovacia mnozina
int alg_estimation = 10;    						//kolko estimacnych cyklov
int alg_learncycles = 20;							//kolko uciacich cyklov
float alg_mse = 0.0;								//MSE (ak je <> 0.0 => ignoruje pocet cyklov
int alg_classes = 10;								//kolko tried mame k dispozicii
int alg_filter_heredity = 0;						//dedenie filtra vramci iteracii 
int alg_builtin_extractor = 0;						//pevny filter (1) alebo uciaci (0)
short alg_act_function = 0;							//aktivacna funkcia
short alg_use_rbf_patterns = 0;						//pouzitie RBF vystupnej vrstvy
char alg_rbf_patterns[255];							//meno suboru s RBF patternami
int alg_relearn = 10;								//kolko krat sa ma alg pokusit o zlepsenie hypotezy

float *alg_ETA = NULL;								//globalny parameter ucenia
float *alg_MI = NULL;								

char **classifier_filename = NULL;					//nazvy suborov kde je struktura klasifikatora
int **classifier_structure =  NULL;					//struktury WeakCNN
char **dataset_names = NULL;						//mena datasetov (v zavislosti na pocte)

read_file **dataset = NULL;							//smernik na datove mnoziny
long** dataset_index = NULL;						//smernik na indexy prikladov
WeakCNN ***classifier = NULL;						//smernik na klasifikatory (trening)
WeakCNN **network = NULL;							//smernik na klasifikatory - uz naucene (test)
masks *builtinextractors = NULL;					//smernik na pevne masky
rbfpatterns *rbfPAT = NULL;							//smernik na RBF patterny

int current_ensemble = 0;							//aktualny pocet klasifikatorov
typedef struct ensemble {							//struktura pre klasifikatory	
	WeakCNN *classifier;
	ensemble *next;
};
ensemble *pHead = NULL, *pAct = NULL, *pTemp = NULL;

int strong_ensemble = 0;							//silne klasifikatory (konvolucne filtre)
typedef struct s_ensemble {							//struktura pre konvolucne filtre
	WeakCNN *classifier;
	s_ensemble *next;
};
s_ensemble *sHead = NULL, *sAct = NULL, *sTemp = NULL;


int alg_procedure = 0;								//0 - train
													//1 - test ensemble
													//2 - test single (strong) 
													//3 - test (system dosadzovania chromozomov)
int alg_voting_scheme = 0;							

//0 - Dynamic Weighted Voting
//1 - Dynamic Weighted Voting with activation of neurons

//1x - daset weights adjustion before learning
//2x - no adjustion
//10 - 20 - Majority Voting - Learn++ model (composite normalized error)
//11 - 21 - Majority Voting - Learn++ model (normalized error + activation)
//12 - 22 - Majority Voting - Learn++ model (composite normalized error + activation)
//13 - 23 - Majority Voting - (normalized error + activation) 

//30 - Majority Voting - on RBF unit activation

//35 - Majority Voting - on RBF + hypotheses + winner confidence activation
//36 - Majority Voting - on RBF + hypotheses + all confidence activation
//37 - Majority Voting - on RBF + composite hypotheses + winner confidence activation
//38 - Majority Voting - on RBF + composite hypotheses + all confidence activation

//40 - Product Voting - hypotheses + winner activation
//41 - Product Voting - hypotheses + all activation
//42 - Product Voting - composite hypotheses + winner activation
//43 - Product Voting - composite hypotheses + all activation

//50 - Weighted Majority Voting - pseudoloss error function + hypotheses
//51 - Weighted Majority Voting - pseudoloss error function + composite hypotheses
//51 - Weighted Majority Voting - pseudoloss error function + partial composite hypotheses

//60 - 

int alg_loss_function = 0;							//0 - normal 
													//1 - pseudoloss on output
													//2 - pseudoloss on RBF units
float alg_radius = 5;								//radius
float alg_rejection_treshold = 0.0;					//faktor odmitnutia, vsetko co je pod hranicou zamietni 
char *alg_lenghts_file = NULL;							//subor s expertnymi znalostami dlzok

struct classes {	
		int name;									//znacka triedy
		short flag;									//vyskyt triedy
		double factor;								//Z - faktor triedy	
		double preliminary;							//P - predbezny koeficient
		double **preliminar;						//P - predbezny koeficient pre priklady
	} *class_pointer;								//smernik na triedy
	
double EPSILON = 0.0;								//kompozitna chyba
double BETA = 0.0;									//normalizovana kompozitna chyba

int current_classes = 0;							//pocitadlo aktualneho poctu tried, s kt. pracujeme
int *act_classes = NULL;							//pole pocetnosti tried, s kt. pracujeme

FILE *log_screen, *log_samples, *log_hyp, *log_test, *log_report;
													//smernik na subory zapisov 

//****************************
//definicia globalnych funkcii 

bool parameters(void);								//nastavenie cez vstupny dubor
long SSwR (read_file *);							//SSwR ruleta
long SSwoR (read_file *);							//SSwoR ruleta

read_file* open_dataset(char *);					//otvor priklady
void close_dataset();								//zatvor priklady
long* pickup_samples(read_file*, int, int);			//tvorba/vyber prikladov na ucenie
bool search_list(int, int *, int);					//prehladavanie zoznamu na vyskyt hodnoty
bool search_long_list(long, long *, long);

void estimate_cnn(WeakCNN*, read_file*, long*, int);		//estimacia pomocou druhych derivacii
void learn_cnn(WeakCNN*, read_file*, long*, int, int, float &, float &);	//ucenie siete 

void calculate_epsilon(WeakCNN*, read_file*);		//vypocet chyby klasifikatora epsilon 
void calculate_classes(read_file*);					//pocitadlo aktualnych tried

void training_procedure(void);						//trenovanie
void testing_procedure(void);						//testovanie
void strong_testing_procedure(void);				//testovanie strong klasifikatora
void pap_testing_procedure(void);					//testovanie mikroskopickych snimok
void lenght_testing_procedure(void);				//testovanie s dlzkami

void copy_filter(WeakCNN*, WeakCNN*);				//kopirovanie filtra

void add_classifier(int, bool);						//pridanie klasifikatora do ensemblu
void delete_classifier(ensemble *);					//vyradenie klasifikatora z ensemblu

void add_strong_classifier(int);					//pridanie silneho klasifikatora
void delete_strong_classifier(s_ensemble *);		//vyradenie silneho klasifikatora

//****
//main
int _tmain(int argc, _TCHAR* argv[])
{
	srand( (unsigned)time( NULL ) );				//seed generatora vah
	
	//nastavenie pracovneho adresara -> default = data
	if (argc == 2) {
		strcpy(working_directory, argv[1]);
	}
	else {
		strcpy(working_directory, "data");
	}
	strcat(working_directory, "\\");
	
	//otvorenie log suborov 
	char file_samples[255], file_screen[255], file_hyp[255];
	strcpy(file_samples, working_directory);
	strcpy(file_screen, working_directory);
	strcpy(file_hyp, working_directory);
	strcat(file_samples, "log-samples");
	strcat(file_screen, "log-screen");
	strcat(file_hyp, "log-hypotheses");
	
	if( !(log_samples = fopen(file_samples, "at")) || !(log_screen = fopen(file_screen, "at")) || !(log_hyp = fopen(file_hyp, "at"))) { 
		printf ("\n\nError in opening log files !!!");
		exit(0);
	}
	char dbuffer[9], tbuffer[9];		//pomocne premenne na uchovanie datumu a casu 
	_strdate( dbuffer );
	_strtime( tbuffer );	
	fprintf(log_samples, "\n***********************************************************************");
	fprintf(log_screen, "\n***********************************************************************");
	fprintf(log_hyp, "\n***********************************************************************");

	fprintf(log_samples, "\nProcess Started : %s, %s", dbuffer, tbuffer);
	fprintf(log_screen, "\nProcess Started : %s, %s", dbuffer, tbuffer);
	fprintf(log_hyp, "\nProcess Started : %s, %s", dbuffer, tbuffer);
	
	fflush(log_samples);
	fflush(log_screen);
	fflush(log_hyp);
	

//start of learn++

	if (false == parameters()) {
		exit (0);
	}

	//inicializacia datovych mnozin
	dataset = new read_file*[alg_datasets];
	dataset_index = new long*[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		dataset[i] = open_dataset(dataset_names[i]);
	}
	
	//normalizacia po mnozinach na 100 (v pripade ze sa jedna o chromozomy)
	for (int i = 0 ; i < alg_datasets ; i++) {
		
		int tmax = 0;
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			if (tmax < dataset[i]->pattern[j].length) {
				tmax = dataset[i]->pattern[j].length;
			}
		}
		
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			dataset[i]->pattern[j].length = (int)( (double)(dataset[i]->pattern[j].length) * (double)100.0 / (double)tmax); 
		}
	}

	//inicializacia tried - podla pouzitych klasifikatorov
	//zatial napevno index = name 
	class_pointer = new classes[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		class_pointer[i].name = i;
		class_pointer[i].flag = 0;
		class_pointer[i].factor = 0.0;
		class_pointer[i].preliminary = 0.0;

		//alokacia Pc pre kazdy priklad, ktorym sa bude ucit
		class_pointer[i].preliminar = new double*[alg_datasets];
		for (int m = 0 ; m < alg_datasets ; m++) {
			class_pointer[i].preliminar[m] = new double[dataset[m]->patterns];
			for (int n = 0 ; n < dataset[m]->patterns ; n++) {
				class_pointer[i].preliminar[m][n] = 0.0;
			}
		}
	}

	//triedy (zatial je to pole 0)
	act_classes = new int[255];
	for (int i = 0 ; i < 255 ; i++) {
		act_classes[i] = 0;
	}
	
	//testovacie suboru klasifikatorov
	if (alg_procedure == 1) {
		testing_procedure();
	}
	//testovanie silnych klasifikatorov
	else if (alg_procedure == 2) {
		strong_testing_procedure();
	}
	//testovanie mikroskopickych snimok chromozomov
	else if (alg_procedure == 3) {
		pap_testing_procedure();
	}
	//testovanie s korekcnym modulom
	else if (alg_procedure == 4) {
		lenght_testing_procedure();
	}
	//trenovanie suboru klasifikatorov
	else {
		training_procedure();
	}
	
	//ukoncenie
	_strdate( dbuffer );
	_strtime( tbuffer );

	fprintf(log_samples, "\nProcess Ended : %s, %s", dbuffer, tbuffer);
	fprintf(log_screen, "\nProcess Ended : %s, %s", dbuffer, tbuffer);
	fprintf(log_hyp, "\nProcess Ended : %s, %s", dbuffer, tbuffer);

	fclose(log_samples);
	fclose(log_screen);
	fclose(log_hyp);
	
	//dealokacia
	delete rbfPAT;
	delete alg_ETA;	
	delete alg_MI;
	
	for ( int a = 0 ; a < alg_datasets ; a++) {
		delete[] classifier_filename[a];
	}
	delete[] classifier_filename;

	for ( int a = 0 ; a < alg_datasets ; a++) {
		delete[] classifier_structure[a];
	}
	delete[] classifier_structure;
	
	for ( int a = 0 ; a < alg_datasets ; a++) {
		delete[] dataset_names[a];
	}
	delete[] dataset_names;
	
	pTemp = pHead;
	for (int t = 0 ; t < current_ensemble ; t++) {
		pAct = pTemp;
		pTemp = pTemp->next;
		delete pAct->classifier;
		free(pAct);
	}

	sTemp = sHead;
	for (int t = 0 ; t < strong_ensemble ; t++) {
		sAct = sTemp;
		sTemp = sTemp->next;
		delete sAct->classifier;
		free(sAct);
	}

	delete builtinextractors;

	for ( int a = 0 ; a < alg_classes ; a++) {
		for ( int d = 0 ; d < alg_datasets ; d++) {
			delete[] class_pointer[a].preliminar[d];

		}
		delete[] class_pointer[a].preliminar;
	}
	delete[] class_pointer;

	for (int i = 0 ; i < alg_datasets ; i++) {
		delete dataset[i];
	}
	delete[] dataset;


//end of learn++

	return 0;
}

//**********
//trenovanie
void training_procedure() {
	
	printf("\n\n >Learning process");
	fprintf(log_screen, "\n\n >Learning process");
	//***************************
	//cez vsetky datove mnoziny k
	//***************************
	for (int k = 0 ; k < alg_datasets ; k++ ) {
		
		//zistenie aktualneho poctu tried
		calculate_classes(dataset[k]);

		//pocitadlo pokusov opakoveneho ucenia sa vynuluje
		int relearned = 0;
		
		//stav uplneho aucenia composite_beta = 0.0
		int composite_beta_flag = 0;

		//stav naucenia filtracnej casti siete
		int heredity_flag = 0;
		
		//ak je alg_voting_scheme 2 - binarne
		dataset[k]->setup_Htfield(rbfPAT->size_x, rbfPAT->size_y);

		//premenna kde sa pamata pocet prikladov ucenia na datasete k
		//bud je pevny alebo je definovany pomerom
		int dataset_samples = 0;
		if (alg_samples != 0) {
			dataset_samples =  alg_samples;
		}
		else {
			dataset_samples = (int)(dataset[k]->patterns * alg_part_samples);
		}

		printf("\n\nDataset : %d (%s)", k, dataset_names[k]);
		fprintf(log_screen, "\n\nDataset : %d (%s)", k, dataset_names[k]);
		
		//k = 0 - inicializacia vah datasetu
		if (k == 0 && current_ensemble == 0) {
			//nastavenie vah u prvej mnoziny
			if (alg_loss_function == 0) {
				dataset[k]->reset_weights();
			}
			else {
				dataset[k]->setup_misslabels(current_classes, act_classes);
				dataset[k]->apply_misslabels();
				
			}
		}
		
		//k > 0 - vylepsena cast
		//nastavenie dalsich mnozin na zaklade vysledkov na uz terajsom subore klasifikatorov
		//distribucia prikladov novej datovej mnoziny je uz upravena podla aktualneho stavu suboru
		else {
						
			double evaulate_EPSILON = 0.0;
			double evaulate_BETA = 0.0;
			
			long misclassified = 0;		//pocitadla
			long totalsamples = 0;
			

			//docasna nahrada
			if (alg_loss_function == 0) {
				dataset[k]->reset_weights();
			}
			else {
				dataset[k]->setup_misslabels(current_classes, act_classes);
				dataset[k]->apply_misslabels();
			}

			switch (alg_voting_scheme) {
				//DWV
				case 0:
				//DWV + activation
				case 1:
					{
					//pre vsetky priklady sa vypocitava vola DWV
					//prva cast - spolocne veci pre datovu mnozinu
					
					double *argmax = new double[alg_classes];
					
					//mormalizacia vah klasifikatorov
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
						pTemp = pTemp->next;
					}
					
					//musime docasne nastavit filter
					if (alg_filter_heredity != 0) {
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(false);
							pTemp = pTemp->next;
						}
					}
					//presunuty 2. krok DWV, ktory staci vyratat raz dopredu
					//normalizacny faktor klasifikacnych tried
					for (int a = 0 ; a < alg_classes ; a++) {
								
						class_pointer[a].factor = 0.0;
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						//cez vsetky klasifikatory v current ensemble co vedia klasifikovat triedu i
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].factor += pTemp->classifier->weightN;
							}
							pTemp = pTemp->next;
						}
					}

					//cez aktualny dataset !!!
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {

							//1. inicializacia vah klasifikatorov - classifier[k][t]->weight
							//cez vsetky klasifikatory v current ensemble co vedia klasifikovat triedu i
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								//reset DWV vah klasifikatorov
								pTemp->classifier->dwvweight = pTemp->classifier->weightN;
								pTemp = pTemp->next;
							}

							//2. Zc - normalizacny faktor pre kazdu triedu
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
								class_pointer[a].preliminary = 0.0;
							}

							//3. predbezny koeficient tried Pc
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								class_pointer[pTemp->classifier->temporal_class].preliminary += pTemp->classifier->dwvweight / class_pointer[pTemp->classifier->temporal_class].factor;
								pTemp = pTemp->next;
							}
							
							//4. updatuju sa vahy vsetkych klasifikatorov 
							//pre tie klasifikatory ktore nepoznaju triedu tohto prikladu upravime vahy na zaklade Pc
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								//test ci dany klasifikator pozna triedu, ak nepozna znizujeme vahu
								for (int a = 0 ; a < alg_classes ; a++) {
									if (false == search_list(a , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
										//pTemp->classifier->dwvweight *= (1.0 - class_pointer[a].preliminary);
										pTemp->classifier->dwvweight *= (1.0 - 0.5*class_pointer[a].preliminary);
									}
								}
								pTemp = pTemp->next;
							}

							//5. vypocet finalnej hypotezy
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {				
								//netreba znovu sirit signal - je to ulozene
								if (alg_voting_scheme == 0) {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight;
								}
								else {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight * ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								pTemp = pTemp->next;
							}
							//vysledna trieda prikladu je max hodnota v poli argmax
							int max_index = 0;			//index triedy
							float max = argmax[0];		//hodnota na vystupe
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									max_index = a;
								}
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = max_index;							
						}
					}

					//povodny stav
					if (alg_filter_heredity != 0) {				
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(true);
							pTemp = pTemp->next;
						}
					}
					delete[] argmax;

					}
					break;
				//Learn++ (kombinacne pravidlo - Majority Voting)
				case 10:
				case 11:
				case 12:
				case 13:
					{
					double *argmax = new double[alg_classes];

					//mormalizacia vah klasifikatorov 
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					double sum_of_composite_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						sum_of_composite_weight += pTemp->classifier->composite_weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
						pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
						pTemp = pTemp->next;
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}
					
					//musime docasne nastavit filter
					if (alg_filter_heredity != 0) {
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(false);
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
							
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								
								argmax[a] = 0;
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								if (alg_voting_scheme == 10) {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN ;//* ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								else {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								pTemp = pTemp->next;
							}
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}

							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = winner;
							
						}
					}

					//povodny stav
					if (alg_filter_heredity != 0) {				
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(true);
							pTemp = pTemp->next;
						}
					}

					delete argmax;
					}
					break;
				//Learn++ (na vystupe distribucne kody / nie vysledne triedy)
				case 30:
					{
					
					//mormalizacia vah klasifikatorov
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
						pTemp = pTemp->next;
					}
					
					//musime docasne nastavit filter
					if (alg_filter_heredity != 0) {
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(false);
							pTemp = pTemp->next;
						}
					}

					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}
				
					//prechod aktualnou mnozinou a ziskanie vystupov z vrsvty
					for (int i = k ; i < k + 1 ; i++) {
						dataset[i]->setup_Htfield(rbfPAT->size_x, rbfPAT->size_y);
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {		
							
							dataset[i]->pattern[j].Htfield = new int*[rbfPAT->size_x];
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								dataset[i]->pattern[j].Htfield[m] = new int[rbfPAT->size_y];
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									dataset[i]->pattern[j].Htfield[m][n] = 0;
								}
							}

							double **binary_output;
							binary_output = new double*[current_ensemble];
							for (int t = 0 ; t < current_ensemble ; t++) {
								binary_output[t] = new double[rbfPAT->size_x*rbfPAT->size_y];
								for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
									binary_output[t][s] = 0.0;
								}
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								//pTemp->classifier->get_output(alg_radius);
								for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
									//binary_output[t][s] = pTemp->classifier->get_binary_output(s) * pTemp->classifier->weightN;
									binary_output[t][s] = pTemp->classifier->get_bit_output(s);
								}

								pTemp = pTemp->next;
							}
														
							//v poli binary_output su 1 a -1 nasobene vahami klasifikatorov, v zmysle vahovaneho votingu klasifikatorov
							//vyhrava ten, ktory ma v absolutnej hodnote vyssiu hodnotu
														
							int tmp = 0;
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									double value = 0.0;
									pTemp = pHead;
									for (int t = 0 ; t < current_ensemble ; t++) {
										value += binary_output[t][tmp];
									}
									pTemp = pTemp->next;
									tmp++;
									value /= current_ensemble;
									
									if (value > 0.0) {
										dataset[i]->pattern[j].Htfield[m][n] = 1;
									}
									if (value < 0.0) {
										dataset[i]->pattern[j].Htfield[m][n] = -1;
									}
									
									dataset[i]->pattern[j].Htfieldd[m][n] = value;
								}
							}
								
							for (int t = 0 ; t < current_ensemble ; t++) {
								delete[] binary_output[t];
							}
							delete[] binary_output;
											
						}
					}				

					//povodny stav
					if (alg_filter_heredity != 0) {				
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(true);
							pTemp = pTemp->next;
						}
					}
					}
					break;
				//Learn++  (kombinacne pravidlo - Product)
				case 40:
				case 41:
				case 42:
				case 43:
					{
					
					double *argmax = new double[alg_classes];
					int *argclass = new int[alg_classes];

					for (int a = 0 ; a < alg_classes ; a++) {
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}
					
					//musime docasne nastavit filter
					if (alg_filter_heredity != 0) {
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(false);
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
												
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
								argclass[a] = 0;
							
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								if (alg_voting_scheme == 40 || alg_voting_scheme == 42) {
									if (argmax[pTemp->classifier->temporal_class] == 0.0) {
										argmax[pTemp->classifier->temporal_class] += ((double)1.0 / pTemp->classifier->temporal_activation);
									}
									else {
										argmax[pTemp->classifier->temporal_class] *= ((double)1.0 / pTemp->classifier->temporal_activation);
									}
								}
								//kompletna aktivacia na vystupe
								else {						
									for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
										int membership = pTemp->classifier->p_OUT->plane[p].membership;
										float activation = pTemp->classifier->p_OUT->plane[p].x;
										if (argmax[membership] == 0.0) {
											argmax[membership] += ((double)1.0 / activation);
										}
										else {
											argmax[membership] *= ((double)1.0 / activation);
										}
									}
								}
								//argmax[pTemp->classifier->temporal_class] *= pTemp->classifier->temporal_activation;
								pTemp = pTemp->next;
							}

							//vysledna trieda prikladu je max hodnota v poli argmax
							//pre klasicke vystupy hladame maximum
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = winner;
						}
					}

					//povodny stav
					if (alg_filter_heredity != 0) {				
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							pTemp->classifier->apply_heredity(true);
							pTemp = pTemp->next;
						}
					}
					delete[] argmax;
					delete[] argclass;

					}
					break;
				
			}
			
			//vypocet pomocneho EPSILON a BETA
			//ak su na vystupe distribucne kody
			if (alg_voting_scheme == 30) {
				double *output;
				output = new double[rbfPAT->number_of_pat];
				
				for (int i = k ; i < k + 1 ; i++) {
					for (int j = 0 ; j < dataset[i]->patterns ; j++) {
						
												
						for (int s = 0 ; s < rbfPAT->number_of_pat ; s++) {
							output[s] = 0.0;
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									//vzialenost vystupu od patternu
									if (alg_voting_scheme == 30) {
										output[s] += (dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n]);
									}
									else {
										output[s] += (dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n]);
									}
								}
							}
						}
						
						double min = output[0];
						int choosen = 0;
						for (int s = 1 ; s < rbfPAT->number_of_pat ; s++) {
							if ( min > output[s]) {
								choosen = s;
								min = output[s];
							}
						}
						dataset[i]->pattern[j].Ht = choosen;

						if (dataset[i]->pattern[j].membership != choosen) {
							evaulate_EPSILON += dataset[i]->pattern[j].weightN;
						}
						
					}
				}
				delete[] output;
			}
			//ak su na vystupe triedy
			else {
				//vypocet EPSILON a BETA
				for (int i = k ; i < k + 1 ; i++) {
					for (int j = 0 ; j < dataset[i]->patterns ; j++) {			
						if (dataset[i]->pattern[j].membership != dataset[i]->pattern[j].Ht) {
							evaulate_EPSILON += dataset[i]->pattern[j].weightN;						
						}
					}
				}
			}
			
			//udrzujeme stale distribuciu tak aby suma vah bola vzdy = 1 pri novom datasete
			//neupravujeme vahy v tychto pripadoch
			if (alg_voting_scheme == 20 ||
				alg_voting_scheme == 21 ||
				alg_voting_scheme == 22 ||
				alg_voting_scheme == 23 ||
				alg_voting_scheme == 35 ||
				alg_voting_scheme == 50 ||
				alg_voting_scheme == 60
				) {
					dataset[k]->normalize_weights();
			}
			//uprava vah v ostatnych pripadoch
			else {
					evaulate_BETA = evaulate_EPSILON / ((double)1.0 - evaulate_EPSILON);
					
					//nastavenie vah prikladov
					for (int i = 0 ; i < dataset[k]->patterns ; i++) {
						if (dataset[k]->pattern[i].Ht == dataset[k]->pattern[i].membership) {
							dataset[k]->pattern[i].weight *= evaulate_BETA;
						}
					}
					dataset[k]->normalize_weights();
			}
		}

		//************************************************
		//vytvaranie klasifikatorov t na datovej mnozine k
		//************************************************
		for (int t = 0 ; t < alg_iterations ; t++) {
			
			//test opakovaneho ucenia, ak algoritmus zlyha alg_relearn krat ucenie na aktualnej mnozine k sa zastavi
			if (relearned > alg_relearn) {
				printf("\nLearning on dataset (%d) finished after %d attempts to improve current hypothesis !", k, relearned-1);
				fprintf(log_screen, "\nLearning on dataset (%d) finished after %d attempts to improve current hypothesis !", k, relearned-1);
				break;
			}
			
			//Ak je nastavene zdielanie filtra, kazdy dalsi klasifikator t pre datovu mnozinu k
			//bude mat rovnaku filtracnu cast (kapitola 5.2.2)
			if (alg_filter_heredity == 0) {
				printf("\nFilter heredity OFF");
				fprintf(log_screen, "\nFilter heredity OFF");
			}
			else {
				printf("\nFilter heredity ON - learning ...");
				fprintf(log_screen, "\nFilter heredity ON - learning ...");
				
				//ak je nastavene zdielanie filtra, najprv sa filter uci na celej mnozine -> ide o kvalitu
				if (heredity_flag == 0) {
					
					//pomocna premenna kde sa ulozia indexy vsetkych prikladov na trening silneho klasifikatora
					long *full_index = new long[dataset[k]->patterns];
					for (int i = 0 ; i < dataset[k]->patterns ; i++) {
						full_index[i] = i; 
					}
					//pridanie struktury kde bude ulozeny nauceny konvolucny filter
					add_strong_classifier(k);
					
					char file_number[5];
					char file_name[255];

					float sMSE = 0.0;		//stredna kvadraticka chyba
					float sBPE = 0.0;		//velkost zmeny vah v sieti

					for (int i = 0 ;  i < alg_filter_heredity ; i++) {
						estimate_cnn(sAct->classifier, dataset[k], full_index, dataset[k]->patterns);
						learn_cnn(sAct->classifier, dataset[k], full_index, dataset[k]->patterns, i, sMSE, sBPE);
						//used_cycles++;
						printf("\n-->Cycle %d, MSE : %f, BPE : %f", i, sMSE, sBPE);
						fprintf(log_screen, "\n-->Cycle %d, MSE : %f, BPE : %f", i, sMSE, sBPE);
						fflush(log_screen);
						sAct->classifier->save_net("data\\strong.net");
					}
					//ulozenie klasifikatora
					//ukladanie klasifikatorov
					sTemp = sHead;
					for (int s = 0 ; s < strong_ensemble ; s++) {
						itoa(s, file_number, 10);
						strcpy(file_name, working_directory);
						strcat(file_name, "scnn");
						strcat(file_name,file_number);
						strcat(file_name,".net");
						sTemp->classifier->save_net(file_name);
						sTemp = sTemp->next;
					}

					//alokujeme pamat pre aktualny novo pridany dataset
					dataset[k]->setup_convolution(strong_ensemble, sAct->classifier->filter_size);
					//nastavime vystup filtra pre vsetky klasifikatory 
						sTemp = sHead;
						for (int s = 0 ; s < strong_ensemble ; s++) {
							for (int i = 0 ; i < dataset[k]->patterns ; i++) {
								sTemp->classifier->assign_input(dataset[k], i);
								sTemp->classifier->feed_forward();
								//sTemp->classifier->get_output(alg_radius);
								for (int j = 0 ; j < sTemp->classifier->filter_size ; j++) {
									dataset[k]->pattern[i].convolution_filter[s][j] = sTemp->classifier->filter_output[j]; 
								}
							}
							sTemp = sTemp->next;
						}
					//vsetky klasifikatory v ensembli budu vyuzivat tento filter na danej mnozine
					heredity_flag = 1;
		
					printf("\n\nDone, continue on ensemble learning");
					fprintf(log_screen, "\n\nDone, continue on ensemble learning");
					
					delete[] full_index;
				}
			}
				
			//alokacia klasifikatora (vytvaranie suboru pridavanim klasifikatorov)
			if (t == 0) {
				add_classifier(k, true);
			}
			else {
				add_classifier(k, false);
			}
			
			//skopiruje sa vzdy prvy filter vramci vsetkych iteracii
			if (heredity_flag != 0) {
				copy_filter(sAct->classifier, pAct->classifier);
				pAct->classifier->apply_heredity(true);
			}

			printf("\n\nIteration : %d", t);
			fprintf(log_screen, "\n\nIteration : %d", t);

			//vyber prikladov na zaklade distribucie ruletovou metodou
			dataset_index[k] = pickup_samples(dataset[k], dataset_samples, 0);
			
			//zapis trenovacich vzorov s ich aktualnymi vahami a tych, kt. oznacila ruleta
			fprintf(log_samples, "\nDataset %d, Iteration %d", k, t);
			fprintf(log_samples, "\nPattern\tMembership\tWeight\tSelection");
			for (int i = 0 ; i < dataset[k]->patterns ; i++) {
				if (true == search_long_list(i, dataset_index[k], dataset_samples)) {
					fprintf(log_samples, "\n%d\t%d\t%f\t*", i, dataset[k]->pattern[i].membership, dataset[k]->pattern[i].weight);
				}
				else {
					fprintf(log_samples, "\n%d\t%d\t%f\t", i, dataset[k]->pattern[i].membership, dataset[k]->pattern[i].weight);
				}
			}
						
			//2. ucenie klasifikatora
			char file_number[5];
			char file_name[255];
			float MSE = 0.0;		//stredna kvadraticka chyba
			float BPE = 0.0;		//velkost zmeny vah v sieti
			int used_cycles = 0;	

			//cez cykly
			if (alg_mse == 0.0) {
				for (int i = 0 ;  i < alg_learncycles ; i++) {
					estimate_cnn(pAct->classifier, dataset[k], dataset_index[k], dataset_samples);
					learn_cnn(pAct->classifier, dataset[k], dataset_index[k], dataset_samples, i, MSE, BPE);
					used_cycles++;
					printf("\n-->Cycle %d, MSE : %f, BPE : %f", i, MSE, BPE);
					fprintf(log_screen, "\n-->Cycle %d, MSE : %f, BPE : %f", i, MSE, BPE);
					fflush(log_screen);
					//classifier[k][t]->save_net("data\\learn.net");
				}
			}
			//na chybu MSE
			else {
				do {
					//if (used_cycles % 50 == 0)
						estimate_cnn(pAct->classifier, dataset[k], dataset_index[k], dataset_samples);
					
					learn_cnn(pAct->classifier, dataset[k], dataset_index[k], dataset_samples, used_cycles, MSE, BPE);
					printf("\n-->Cycle %d, MSE : %f, BPE : %f", used_cycles, MSE, BPE);
					fprintf(log_screen, "\n-->Cycle %d, MSE : %f, BPE : %f", used_cycles, MSE, BPE);
					fflush(log_screen);
					//pAct->classifier->save_net("data\\learn.net");
					used_cycles++;
					
				} while (alg_mse < MSE && alg_learncycles > used_cycles);
			}

			//3. hypoteza ht, vypocet epsilon a beta
			calculate_epsilon(pAct->classifier, dataset[k]);
			
			//ak je epsilon > 1/2 opakovat !!!	- klasifikator nesplnil kriterium
			if (pAct->classifier->epsilon >= 0.5) {
				
				printf("\nCurrent hypothesis is too weak, misclassification %d, epsilon %f !", pAct->classifier->misclassification, pAct->classifier->epsilon);
				fprintf(log_screen, "\nCurrent hypothesis is too weak, misclassification %d, epsilon %f !", pAct->classifier->misclassification, pAct->classifier->epsilon);
				
				delete_classifier(pAct);
				
				//add_classifier(k);
				
				relearned++;
				t--;
				continue;
			}
			
			//relearned = 0;

			//pokracujeme vypocet beta
			pAct->classifier->beta = pAct->classifier->epsilon / ((double)1.0 - pAct->classifier->epsilon);
			
			//nastavenie vahy klasifikatora 
			if (pAct->classifier->beta == 0.0) {
				//nastavime beta na hodnotu prikladu, ktory ma najmensiu distribucnu vahu v datasete k
				double minimum = dataset[k]->pattern[0].weightN;
				for (int i = 1 ; i < dataset[k]->patterns ; i++) {
					if (minimum > dataset[k]->pattern[i].weightN) {
						minimum = dataset[k]->pattern[i].weightN;
					}
				}
				pAct->classifier->beta = minimum / ((double)1.0 - minimum);
				
			}
			pAct->classifier->weight = log10((double)1.0/pAct->classifier->beta);
			
			//pAct->classifier->save_net("data\\lastone.net");
			
			//4. zistenie tried klasifikatora
			//triedy su ulozene v classifier[k][t]->classes 
			
			//5. Voting scheme - spajanie klasifikatorov
			long totalsamples = 0;

			switch (alg_voting_scheme) {
				//DWV
				case 0:
				//DWV + activation
				case 1:
					{
					//pre vsetky priklady sa vypocitava vola DWV
					//prva cast - spolocne veci pre datovu mnozinu
					
					double *argmax = new double[alg_classes];
					
					//mormalizacia vah klasifikatorov
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
						pTemp = pTemp->next;
					}
					
					//presunuty 2. krok DWV, ktory staci vyratat raz dopredu
					//normalizacny faktor klasifikacnych tried
					for (int a = 0 ; a < alg_classes ; a++) {
								
						class_pointer[a].factor = 0.0;
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						//cez vsetky klasifikatory v current ensemble co vedia klasifikovat triedu i
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].factor += pTemp->classifier->weightN;
							}
							pTemp = pTemp->next;
						}
					}

					//cez aktualny dataset !!!
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {

							//1. inicializacia vah klasifikatorov - classifier[k][t]->weight
							//cez vsetky klasifikatory v current ensemble co vedia klasifikovat triedu i
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								//reset DWV vah klasifikatorov
								pTemp->classifier->dwvweight = pTemp->classifier->weightN;
								pTemp = pTemp->next;
							}

							//2. Zc - normalizacny faktor pre kazdu triedu
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
								class_pointer[a].preliminary = 0.0;
							}

							//3. predbezny koeficient tried Pc
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								class_pointer[pTemp->classifier->temporal_class].preliminary += pTemp->classifier->dwvweight / class_pointer[pTemp->classifier->temporal_class].factor;
								pTemp = pTemp->next;
							}
							
							//4. updatuju sa vahy vsetkych klasifikatorov 
							//pre tie klasifikatory ktore nepoznaju triedu tohto prikladu upravime vahy na zaklade Pc
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								//test ci dany klasifikator pozna triedu, ak nepozna znizujeme vahu
								for (int a = 0 ; a < alg_classes ; a++) {
									if (false == search_list(a , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
										//pTemp->classifier->dwvweight *= (1.0 - class_pointer[a].preliminary);
										pTemp->classifier->dwvweight *= (1.0 - 0.5*class_pointer[a].preliminary);
									}
								}
								pTemp = pTemp->next;
							}

							//5. vypocet finalnej hypotezy
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {				
								//netreba znovu sirit signal - je to ulozene
								if (alg_voting_scheme == 0) {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight;
								}
								else {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight * ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								pTemp = pTemp->next;
							}
							//vysledna trieda prikladu je max hodnota v poli argmax
							int max_index = 0;			//index triedy
							float max = argmax[0];		//hodnota na vystupe
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									max_index = a;
								}
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = max_index;							
						}
					}
					delete[] argmax;
					}
					break;
				//Learn++ (majority voting)
				case 10:
				case 20:
				case 11:
				case 21:
				case 12:
				case 22:
				case 13:
				case 23:
					{
					double *argmax = new double[alg_classes];

					//mormalizacia vah klasifikatorov 
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					double sum_of_composite_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						sum_of_composite_weight += pTemp->classifier->composite_weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
						pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
						pTemp = pTemp->next;
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
							
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								
								argmax[a] = 0;
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								if (alg_voting_scheme == 10 || alg_voting_scheme == 20) {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN ;//* ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								else {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
								}
								pTemp = pTemp->next;
							}
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}

							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = winner;
							
						}
					}
					delete[] argmax;
					}
					break;
				//Learn++ (vystupna vrstva distribucnych kodov)
				case 30:
					{
					
					//mormalizacia vah klasifikatorov
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
						pTemp = pTemp->next;
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}
				
					//prechod aktualnou mnozinou a ziskanie vystupov z vrsvty
					for (int i = k ; i < k + 1 ; i++) {
						dataset[i]->setup_Htfield(rbfPAT->size_x, rbfPAT->size_y);
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {		
							
							dataset[i]->pattern[j].Htfield = new int*[rbfPAT->size_x];
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								dataset[i]->pattern[j].Htfield[m] = new int[rbfPAT->size_y];
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									dataset[i]->pattern[j].Htfield[m][n] = 0;
								}
							}

							double **binary_output;
							binary_output = new double*[current_ensemble];
							for (int t = 0 ; t < current_ensemble ; t++) {
								binary_output[t] = new double[rbfPAT->size_x*rbfPAT->size_y];
								for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
									binary_output[t][s] = 0.0;
								}
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								//pTemp->classifier->get_output(alg_radius);
								for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
									//binary_output[t][s] = pTemp->classifier->get_binary_output(s) * pTemp->classifier->weightN;
									binary_output[t][s] = pTemp->classifier->get_bit_output(s);
								}

								pTemp = pTemp->next;
							}
														
							//v poli binary_output su 1 a -1 nasobene vahami klasifikatorov, v zmysle vahovaneho votingu klasifikatorov
							//vyhrava ten, ktory ma v absolutnej hodnote vyssiu hodnotu
														
							int tmp = 0;
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									double value = 0.0;
									pTemp = pHead;
									for (int t = 0 ; t < current_ensemble ; t++) {
										value += binary_output[t][tmp];
									}
									pTemp = pTemp->next;
									tmp++;
									value /= current_ensemble;
									
									if (value > 0.0) {
										dataset[i]->pattern[j].Htfield[m][n] = 1;
									}
									if (value < 0.0) {
										dataset[i]->pattern[j].Htfield[m][n] = -1;
									}
									
									dataset[i]->pattern[j].Htfieldd[m][n] = value;
								}
							}
								
							for (int t = 0 ; t < current_ensemble ; t++) {
								delete[] binary_output[t];
							}
							delete[] binary_output;
						}
					}
					}
					break;
				//Learn++ -> Majority Voting on RBF confidence activation
				case 35:
				case 36:
				case 37:
				case 38:
					{
					
					double *argmax = new double[alg_classes];
					int *argclass = new int[alg_classes];

					for (int a = 0 ; a < alg_classes ; a++) {
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
												
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
								argclass[a] = 0;
							
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward();
								pTemp->classifier->get_output(alg_radius);
								pTemp->classifier->get_confidence(alg_radius);
								if (alg_voting_scheme == 35 || alg_voting_scheme == 37) {
									argmax[pTemp->classifier->temporal_class] += pTemp->classifier->temporal_confidence;
									
								}
								else {
									//kompletna aktivacia na vystupe
									for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
										int membership = pTemp->classifier->p_OUT->plane[p].membership;
										float confidence = pTemp->classifier->p_OUT->plane[p].conf;
										argmax[membership] += confidence;
										
									}
								}
								//argmax[pTemp->classifier->temporal_class] *= pTemp->classifier->temporal_activation;
								pTemp = pTemp->next;
							}

							//vysledna trieda prikladu je max hodnota v poli argmax
							//pre klasicke vystupy hladame maximum
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = winner;
						}
					}
					delete[] argmax;
					delete[] argclass;
					}
					break;

				//Learn++  (Product Voting)
				case 40:
				case 41:
				case 42:
				case 43:
					{
					
					double *argmax = new double[alg_classes];
					int *argclass = new int[alg_classes];

					for (int a = 0 ; a < alg_classes ; a++) {
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
												
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
								argclass[a] = 0;
							
							}
							
							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								if (alg_voting_scheme == 40 || alg_voting_scheme == 42) {
									if (argmax[pTemp->classifier->temporal_class] == 0.0) {
										argmax[pTemp->classifier->temporal_class] += ((double)1.0 / pTemp->classifier->temporal_activation);
									}
									else {
										argmax[pTemp->classifier->temporal_class] *= ((double)1.0 / pTemp->classifier->temporal_activation);
									}
								}
								else {
									//kompletna aktivacia na vystupe
									for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
										int membership = pTemp->classifier->p_OUT->plane[p].membership;
										float activation = pTemp->classifier->p_OUT->plane[p].x;
										if (argmax[membership] == 0.0) {
											argmax[membership] += ((double)1.0 / activation);
										}
										else {
											argmax[membership] *= ((double)1.0 / activation);
										}
									}
								}
								//argmax[pTemp->classifier->temporal_class] *= pTemp->classifier->temporal_activation;
								pTemp = pTemp->next;
							}

							//vysledna trieda prikladu je max hodnota v poli argmax
							//pre klasicke vystupy hladame maximum
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = winner;
						}
					}
					delete[] argmax;
					delete[] argclass;
					}
					break;
			//Learn++ (pseudoloss hypotezy)
				case 50:
				case 51:
				case 52:
					{
					double *argmax = new double[alg_classes];

					//mormalizacia vah klasifikatorov 
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					double sum_of_composite_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						sum_of_composite_weight += pTemp->classifier->composite_weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
						pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
						pTemp = pTemp->next;
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
							
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								
								argmax[a] = 0;
							}
							if (dataset[i]->pattern[j].Hconf == NULL) {
								dataset[i]->pattern[j].Hconf = new double[alg_classes];
							}
							for (int a = 0 ; a < alg_classes ; a++) {
								dataset[i]->pattern[j].Hconf[a] = 0.0;
							}

							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								pTemp->classifier->assign_input(dataset[i], j);
								pTemp->classifier->feed_forward(); 
								pTemp->classifier->get_output(alg_radius);
								pTemp->classifier->get_confidence(alg_radius);
								
								double *prnt = new double[alg_classes];
								double *act = new double[alg_classes];
								for (int p = 0 ; p < alg_classes ; p++) {
									prnt[p] = 0.0;
									act[p] = 0.0;
								}
								//kompletna aktivacia na vystupe
								for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
									int membership = pTemp->classifier->p_OUT->plane[p].membership;
									float confidence = pTemp->classifier->p_OUT->plane[p].conf;
									prnt[membership] = confidence;
									act[membership] = pTemp->classifier->p_OUT->plane[p].x;
									dataset[i]->pattern[j].Hconf[membership] += pTemp->classifier->weightN * confidence;
								}
								

								delete[] prnt;
								delete[] act;
								pTemp = pTemp->next;
							}
							
							//aplikacia SOFTMAX na ziskanie confidence
							double sum = 0.0;
							for (int a = 0 ; a < alg_classes ; a++) {
								//dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
								dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
								sum += dataset[i]->pattern[j].Hconf[a];
							}
							
							for (int a = 0 ; a < alg_classes ; a++) {
								dataset[i]->pattern[j].Hconf[a] /= sum;
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = 0;
							float max = dataset[i]->pattern[j].Hconf[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (dataset[i]->pattern[j].Hconf[a] > max) {
									max = dataset[i]->pattern[j].Hconf[a];
									dataset[i]->pattern[j].Ht = a;
								}
							}
							
						}
					}
					delete[] argmax;
					}
					break;
			
			//Learn++  (pseudoloss kompozitne hypotezy  
				case 60:
					{
					double *argmax = new double[alg_classes];

					//mormalizacia vah klasifikatorov 
					//spocitaju sa vsetky vahy
					double sum_of_weight = 0.0;
					double sum_of_composite_weight = 0.0;
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						sum_of_weight += pTemp->classifier->weight;
						sum_of_composite_weight += pTemp->classifier->composite_weight;
						pTemp = pTemp->next;
					}
					
					//prenastavenie vah
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
						pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
						pTemp = pTemp->next;
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
								
						//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
						pTemp = pHead;
						for (int t = 0 ; t < current_ensemble ; t++) {
							if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								class_pointer[a].flag++;
							}
							pTemp = pTemp->next;
						}
					}

					//test na aktualnom datasete
					for (int i = k ; i < k + 1 ; i++) {
						for (int j = 0 ; j < dataset[i]->patterns ; j++) {
							
							//reset pomocnych premennych
							for (int a = 0 ; a < alg_classes ; a++) {
								
								argmax[a] = 0;
							}
							if (dataset[i]->pattern[j].Hconf == NULL) {
								dataset[i]->pattern[j].Hconf = new double[alg_classes];
							}
							for (int a = 0 ; a < alg_classes ; a++) {
								dataset[i]->pattern[j].Hconf[a] = 0.0;
							}

							//cez vsetky klasifikatory
							pTemp = pHead;
							for (int t = 0 ; t < current_ensemble ; t++) {
								
								//if (pTemp->classifier->dataset == i) {
									pTemp->classifier->assign_input(dataset[i], j);
									pTemp->classifier->feed_forward(); 
									pTemp->classifier->get_output(alg_radius);
									pTemp->classifier->get_confidence(alg_radius);
									
									double *prnt = new double[alg_classes];
									double *act = new double[alg_classes];
									for (int p = 0 ; p < alg_classes ; p++) {
										prnt[p] = 0.0;
										act[p] = 0.0;
									}
									//kompletna aktivacia na vystupe
									for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
										int membership = pTemp->classifier->p_OUT->plane[p].membership;
										float confidence = pTemp->classifier->p_OUT->plane[p].conf;
										prnt[membership] = confidence;
										act[membership] = pTemp->classifier->p_OUT->plane[p].x;
										dataset[i]->pattern[j].Hconf[membership] += pTemp->classifier->weightN * confidence;
									}
									

									delete[] prnt;
									delete[] act;
								//}

								pTemp = pTemp->next;
							}
							
							//aplikacia SOFTMAX na ziskanie confidence
							double sum = 0.0;
							for (int a = 0 ; a < alg_classes ; a++) {
								if (dataset[i]->pattern[j].Hconf[a] != 0.0) {
									dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
								}
								//dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
								sum += dataset[i]->pattern[j].Hconf[a];
							}
							
							for (int a = 0 ; a < alg_classes ; a++) {
								dataset[i]->pattern[j].Hconf[a] /= sum;
							}
							
							//docasne ulozenie Ht pre dany priklad
							dataset[i]->pattern[j].Ht = 0;
							float max = dataset[i]->pattern[j].Hconf[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (dataset[i]->pattern[j].Hconf[a] > max) {
									max = dataset[i]->pattern[j].Hconf[a];
									dataset[i]->pattern[j].Ht = a;
								}
							}

							//docasne ulozenie Ht pre dany priklad
							
						}
					}
					delete[] argmax;
					}
					break;
			}
			
			if (alg_voting_scheme == 30) {
				//vypocet EPSILON a BETA
				double *output;
				output = new double[rbfPAT->number_of_pat];

				for (int i = k ; i < k + 1 ; i++) {
					for (int j = 0 ; j < dataset[i]->patterns ; j++) {
						
						for (int s = 0 ; s < rbfPAT->number_of_pat ; s++) {
							output[s] = 0.0;
							for (int m = 0 ; m < rbfPAT->size_x ; m++) {
								for (int n = 0 ; n < rbfPAT->size_y ; n++) {
									//vzialenost vystupu od patternu
									
									if (alg_voting_scheme == 30) {
										output[s] += (dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n]);
									}
									else {
										output[s] += (dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n]);
									}
									
									
								}
							}
						}
						
						double min = output[0];
						int choosen = 0;
						for (int s = 1 ; s < rbfPAT->number_of_pat ; s++) {
							if ( min > output[s]) {
								choosen = s;
								min = output[s];
							}
						}
						dataset[i]->pattern[j].Ht = choosen;
						if (dataset[i]->pattern[j].membership != choosen) {
							pAct->classifier->composite_epsilon += dataset[i]->pattern[j].weightN;
							pAct->classifier->composite_misclassification++;
							
						}
						totalsamples++;
					}
				}
			}
			else {
				//vypocet EPSILON a BETA
								
				for (int i = k ; i < k + 1 ; i++) {
					for (int j = 0 ; j < dataset[i]->patterns ; j++) {
						
						//klasika
						if (alg_loss_function == 0) {
							if (dataset[i]->pattern[j].membership != dataset[i]->pattern[j].Ht) {
								pAct->classifier->composite_epsilon += dataset[i]->pattern[j].weightN;
								pAct->classifier->composite_misclassification++;
							}
						}
						else if (alg_loss_function == 1) {

						}
						else {
							
							if (dataset[i]->pattern[j].membership != dataset[i]->pattern[j].Ht) {
								pAct->classifier->composite_misclassification++;
							}
							
							//zratava sa chyba epsilon cez mnozinu misslabels
							double pseudo = 0.0;
							for (int a = 0 ; a < alg_classes ; a++) {
								pseudo += dataset[i]->pattern[j].misslabelN_glo[a] * ((double)1.0 - dataset[i]->pattern[j].Hconf[dataset[i]->pattern[j].membership] + dataset[i]->pattern[j].Hconf[a]);
								
							}
							pAct->classifier->composite_epsilon  += (double)0.5 * pseudo;
						}

						totalsamples++;
					}
				}
			}
			
			//if (alg_loss_function == 0) {
				//ak je EPSILON > 0.5 tak opakovat vyber
				if (pAct->classifier->composite_epsilon >= 0.5) {
					
					printf("\nCurrent composite hypothesis is too weak, misclassification %d, epsilon %f !", pAct->classifier->composite_misclassification++, pAct->classifier->composite_epsilon);
					fprintf(log_screen, "\nCurrent composite hypothesis is too weak, misclassification %d, epsilon %f !", pAct->classifier->composite_misclassification, pAct->classifier->composite_epsilon);
					
					delete_classifier(pAct);

					//add_classifier(k);
		
					relearned++;
					t--;
					continue;
					
				}
			//}
			relearned = 0;

			pAct->classifier->composite_beta = pAct->classifier->composite_epsilon / ((double)1.0 - pAct->classifier->composite_epsilon);
			
			if (pAct->classifier->composite_beta == 0.0) {
				//nastavime beta na hodnotu prikladu, ktory ma najmensiu distribucnu vahu v datasete k
				double minimum = dataset[k]->pattern[0].weightN;
				for (int i = 1 ; i < dataset[k]->patterns ; i++) {
					if (minimum > dataset[k]->pattern[i].weightN) {
						minimum = dataset[k]->pattern[i].weightN;
					}
				}
				pAct->classifier->composite_beta = minimum / ((double)1.0 - minimum);
				composite_beta_flag = 1;
			}
			
			pAct->classifier->composite_weight = log10((double)1.0 / pAct->classifier->composite_beta);
			
			//vypisy na obrazovku 
			//vaha aktualneho klasifikatora
			printf("\n-->Current classifier misclassification: %d (%d)", pAct->classifier->misclassification, dataset[k]->patterns);
			printf("\n-->Current classifier weight W(%d): %e", current_ensemble-1, pAct->classifier->weight);
			printf("\n-->Current classifier error epsilon(%d): %e", current_ensemble-1, pAct->classifier->epsilon);
			printf("\n-->Current classifier normalized error beta(%d): %e", current_ensemble-1, pAct->classifier->beta);
			
			fprintf(log_screen, "\n-->Current classifier misclassification: %d (%d)", pAct->classifier->misclassification, dataset[k]->patterns);
			fprintf(log_screen, "\n-->Current classifier weight W(%d): %e", current_ensemble-1, pAct->classifier->weight);
			fprintf(log_screen, "\n-->Current classifier error epsilon(%d): %e", current_ensemble-1, pAct->classifier->epsilon);
			fprintf(log_screen, "\n-->Current classifier normalized error beta(%d): %e", current_ensemble-1, pAct->classifier->beta);

			//vypis vah klasifikatorov
			int poc = 0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				printf("\n-->Classifier (%d) weight [composite]: %e [%e]", poc, pTemp->classifier->weight, pTemp->classifier->composite_weight);
				fprintf(log_screen, "\n-->Classifier (%d) weight [composite]: %e [%e]", poc, pTemp->classifier->weight, pTemp->classifier->composite_weight);
				poc++;
				pTemp = pTemp->next;
			}
			
			//aktualny stav kompozitnej hypotezy
			printf("\n");
			printf("\n-->Composite misclassification: %d (%d)", pAct->classifier->composite_misclassification, totalsamples);
			printf("\n-->Composite weight W(%d): %e", current_ensemble-1, pAct->classifier->composite_weight);
			printf("\n-->Composite error E(%d): %e", current_ensemble-1, pAct->classifier->composite_epsilon);
			printf("\n-->Composite normalized error B(%d): %e", current_ensemble-1, pAct->classifier->composite_beta);
			
			fprintf(log_screen, "\n");
			fprintf(log_screen, "\n-->Composite misclassification: %d (%d)", pAct->classifier->composite_misclassification, totalsamples);
			fprintf(log_screen, "\n-->Composite weight W(%d): %e", current_ensemble-1, pAct->classifier->composite_weight);
			fprintf(log_screen, "\n-->Composite error E(%d): %e", current_ensemble-1, pAct->classifier->composite_epsilon);
			fprintf(log_screen, "\n-->Composite normalized error B(%d): %e", current_ensemble-1, pAct->classifier->composite_beta);
			fflush(log_screen);
			
			//vypis do suboru hypotez
			fprintf(log_hyp, "\n");
			fprintf(log_hyp, "%d\t", k);					//dataset k
			fprintf(log_hyp, "%d\t", t);					//iteracia (hypoteza) t
			fprintf(log_hyp, "%d\t", used_cycles);			//pocet uciacych cyklov
			fprintf(log_hyp, "%f\t", MSE);					//dosiahnute MSE
			fprintf(log_hyp, "%d\t", pAct->classifier->misclassification);		//nespravne klasifikovane hypotezou t
			fprintf(log_hyp, "%d\t", dataset[k]->patterns);						//pocet prikladov v datasete k
			fprintf(log_hyp, "%e\t", pAct->classifier->weight);					//vaha vytvoreneho klasifikatora
			fprintf(log_hyp, "%e\t", pAct->classifier->epsilon);				//hypoteza klasifikatora
			fprintf(log_hyp, "%e\t", pAct->classifier->beta);					//chyba klasifikatora
			fprintf(log_hyp, "%d\t", pAct->classifier->composite_misclassification);	//nespravne klasifikovane kompozitnou hypotezou T
			fprintf(log_hyp, "%d\t", totalsamples);										//pocet prikladov v datasete k
			fprintf(log_hyp, "%e\t", pAct->classifier->composite_weight);				//kompozitna vaha suboru klasifikatorov
			fprintf(log_hyp, "%e\t", pAct->classifier->composite_epsilon);				//kompozitna hypoteza klasifikatora
			fprintf(log_hyp, "%e\t", pAct->classifier->composite_beta);					//kompozitna chyba klasifikatora
			fflush(log_hyp);

			//***************************************************************************	
			
			//ulozenie klasifikatora - ocislovanie je zaradom (k*tmax+t)
			//ukladanie klasifikatorov podla iteracii + berie sa do uvahy current ensemble
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				itoa(t, file_number, 10);
				strcpy(file_name, working_directory);
				strcat(file_name, "cnn");
				strcat(file_name,file_number);
				strcat(file_name,".net");
				pTemp->classifier->save_net(file_name);
				pTemp = pTemp->next;
			}
			
			//6. kompozitna hypoteza Ht, EPSILON, BETA
			//presunuty vypocet do DMW
	
			//classifier_counter++;

			//ak sa dosiahla chyba 0 - vsetko klasifikovane pokracuje sa na dalsej datovej mnozine
			if (composite_beta_flag != 0) {
				break;
			}

			//7. nastavenie vah prikladov
			if (alg_loss_function == 0) {		
				for (int i = 0 ; i < dataset[k]->patterns ; i++) {
					if (dataset[k]->pattern[i].Ht == dataset[k]->pattern[i].membership) {
						dataset[k]->pattern[i].weight *= pAct->classifier->composite_beta;
					}
				}
			
				//udrzujeme stale distribuciu tak aby suma vah bola vzdy = 1
				dataset[k]->normalize_weights();
			}
			else {
				for (int i = 0 ; i < dataset[k]->patterns ; i++) {
					for (int a = 0 ; a < alg_classes ; a++) {
						double exponent = (double)0.5 * ((double)1.0 + dataset[k]->pattern[i].Hconf[dataset[k]->pattern[i].membership] - dataset[k]->pattern[i].Hconf[a]);
						dataset[k]->pattern[i].misslabel[a] *= pow(pAct->classifier->composite_beta, exponent);
					}
				}
				//normalizacie
				dataset[k]->global_misslabels();
				dataset[k]->local_misslabels(current_classes, act_classes);
				dataset[k]->apply_misslabels();
			}

		}

	}
	//finalna hypoteza Hfinal
}

//***********
//testovanie
void testing_procedure() {
	
	int total = 0, total_counter = 0, *dataset_counter = new int[alg_datasets];
	float total_accuracy = 0.0, *dataset_accuracy = new float[alg_datasets];
	char file_name[255];
	
	strcpy(file_name, working_directory);
	
	//docasna premenna pre kompozitne hypotezy
	int *composite_hypothesis = new int[current_ensemble];
	double *composite_hypothesis_value = new double[current_ensemble];
	
	//pocet skupin klasifikatorov je x + 1, kde x je dataset posledneho v ensembli
	pTemp = pHead;
	while (pTemp->next != NULL) {
		pTemp = pTemp->next;
	}
	int learnt_datasets = pTemp->classifier->dataset + 1;

	//pocty klasifikatorov v skupinach podla datasetov
	int *dataset_classifiers = new int[learnt_datasets];
	for (int i = 0 ; i < learnt_datasets ; i++) {
		dataset_classifiers[i] = 0;
	}
	pTemp = pHead;
	for (int t = 0 ; t < current_ensemble ; t++) {
		dataset_classifiers[pTemp->classifier->dataset]++;
		pTemp = pTemp->next;
	}
	//docasna premenna pre finalnu hypotezu;
	int **final_hypothesis = new int*[learnt_datasets];
	for (int i = 0 ; i < learnt_datasets ; i++) {
		final_hypothesis[i] = new int[dataset_classifiers[i]];
		for (int j = 0 ; j < dataset_classifiers[i] ; j++) {
			final_hypothesis[i][j] = -1;
		}
	}

	//docasna premenna pre finalnu hypotezu;
	double **dfinal_hypothesis = new double*[learnt_datasets];
	for (int i = 0 ; i < learnt_datasets ; i++) {
		dfinal_hypothesis[i] = new double[dataset_classifiers[i]];
		for (int j = 0 ; j < dataset_classifiers[i] ; j++) {
			dfinal_hypothesis[i][j] = -1.0;
		}
	}
	
	//docasna premenna pre argmax Hfinal
	double *argmax = new double[alg_classes];
	int *argclass = new int[alg_classes];
	
	//konfidencna tabulka pre priklady
	double **confidence = new double*[alg_classes];
	long total_samples = 0;
	for (int i = 0 ; i < alg_datasets ; i++) {
		total_samples += dataset[i]->patterns;
	}
	for (int i = 0 ; i < alg_classes ; i++) {
		confidence[i] = new double[total_samples];
		for (int j = 0 ; j < total_samples ; j++) {
			confidence[i][j] = 0.0;
		}
	}

	//komparacna tabluka alg_classes*alg_classes
	int **compare = new int*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		compare[i] = new int[alg_classes];
		for (int j = 0 ; j < alg_classes ; j++) {
			compare[i][j] = 0;
		}
	}

	printf("\n\n >Testing ");
	
	GetConsoleScreenBufferInfo(hStdout, &csbiInfo); 
	coordScreen.X = csbiInfo.dwCursorPosition.X;
	coordScreen.Y = csbiInfo.dwCursorPosition.Y;

	switch (alg_voting_scheme) {
		//DWV
		case 0:
		//DWV + activation
		case 1:
			{
			strcat(file_name, "log-test-0_1");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
				pTemp = pTemp->next;
			}
			
			//presunuty 2. krok, ktory staci vyratat raz dopredu
			for (int a = 0 ; a < alg_classes ; a++) {
						
				class_pointer[a].factor = 0.0;
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].factor += pTemp->classifier->weightN;
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
				fprintf(log_test,"\n%f", class_pointer[a].factor);
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					fprintf(log_test, "\n----------------------------------------------------");
					fprintf(log_test, "\nDataset(%d) Pat(%d)", i, j);
										
					//1. inicializacia vah klasifikatorov - classifier[k][t]->weight
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						//reset DWV vah klasifikatorov
						pTemp->classifier->dwvweight = pTemp->classifier->weightN;
						pTemp = pTemp->next;
					}

					//2. Zc - normalizacny faktor pre kazdu triedu
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
						argclass[a] = 0;
						class_pointer[a].preliminary = 0.0;
					}

					//3. predbezny koeficient tried Pc
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						class_pointer[pTemp->classifier->temporal_class].preliminary += pTemp->classifier->dwvweight / class_pointer[pTemp->classifier->temporal_class].factor;
						pTemp = pTemp->next;
					}
					
					//4. updatuju sa vahy vsetkych klasifikatorov 
					//pre tie klasifikatory ktore nepoznaju triedu tohto prikladu upravime vahy na zaklade Pc
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						//test ci dany klasifikator pozna triedu, ak nepozna znizujeme vahu
						for (int a = 0 ; a < alg_classes ; a++) {
							if (false == search_list(a , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
								pTemp->classifier->dwvweight *= (1.0 - class_pointer[a].preliminary);
								
							}
						}
						fprintf(log_test, "\nNetwork (%d) %f, [%d], [%f]", t, pTemp->classifier->dwvweight, pTemp->classifier->temporal_class, pTemp->classifier->temporal_activation);
						pTemp = pTemp->next;
					}
					fflush(log_test);
					
					//5. vypocet finalnej hypotezy
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						//netreba znovu sirit signal - je to ulozene
						if (alg_voting_scheme == 0) {
							argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight;// * ((double)1.0 / pTemp->classifier->temporal_activation);
							confidence[pTemp->classifier->temporal_class][total] += pTemp->classifier->dwvweight;// * ((double)1.0 / pTemp->classifier->temporal_activation);
						}
						else {
							argmax[pTemp->classifier->temporal_class] += pTemp->classifier->dwvweight * ((double)1.0 / pTemp->classifier->temporal_activation);
							confidence[pTemp->classifier->temporal_class][total] += pTemp->classifier->dwvweight * ((double)1.0 / pTemp->classifier->temporal_activation);
						}
						pTemp = pTemp->next;
					}
					
					
					//vysledna trieda prikladu je max hodnota v poli argmax
					int max_index = 0;			//index triedy
					float max = argmax[0];		//hodnota na vystupe
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							max_index = a;
						}
					}
					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = max_index;
					
					//vypocet konfidencie vystupu z ensemblu
					double confidence_sum = 0.0;
					for (int a = 0 ; a < alg_classes ; a++) {
						if (class_pointer[a].flag > 0) {
							confidence_sum += exp(confidence[a][total]);
						}
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						if (class_pointer[a].flag > 0) {
							confidence[a][total] = exp(confidence[a][total]) / confidence_sum;
						}
					}
					// -----------------------

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
					
				}
			}
			}
			break;
		//Learn++ -> majority voting
		case 10:
		case 20:
			{
			strcat(file_name, "log-test-10_20");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						
						composite_hypothesis[t] = 0;
						
						pTemp = pTemp->next;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								argmax[pE->classifier->temporal_class] += pE->classifier->weightN;// * ((double)1.0 / pE->classifier->temporal_activation);
								pE = pE->next;
							}
							//free(pE);
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}

							final_hypothesis[d][e] = winner;
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}

										
					//finalna hypoteza
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							argmax[final_hypothesis[d][e]] += pTemp->classifier->composite_weightN;// * ((double)1.0 / pTemp->classifier->temporal_activation);
							pTemp = pTemp->next;
						}
					}

					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> majority voting (obycajnych hypotez) + aktivacia
		case 11:
		case 21:
			{
			strcat(file_name, "log-test-11_21");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						
						pTemp = pTemp->next;
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								argmax[pE->classifier->temporal_class] += pE->classifier->weightN * ((double)1.0 / pE->classifier->temporal_activation);
								pE = pE->next;
							}
							//free(pE);
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}

							final_hypothesis[d][e] = winner;
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}
					
					//finalna hypoteza po skupinach
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							//argmax[final_hypothesis[d][e]] += pTemp->classifier->composite_weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
							argmax[final_hypothesis[d][e]] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
							pTemp = pTemp->next;
						}
					}

					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> majority voting (kompozitnych hypotez) + aktivacia
		case 12:
		case 22:
			{
			strcat(file_name, "log-test-12_22");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}

			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
										
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
												
						pTemp = pTemp->next;
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								argmax[pE->classifier->temporal_class] += pE->classifier->weightN * ((double)1.0 / pE->classifier->temporal_activation);
								pE = pE->next;
							}
							//free(pE);
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}

							final_hypothesis[d][e] = winner;
							dfinal_hypothesis[d][e] = max;
							
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}
					
					//finalna hypoteza po skupinach
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							//argmax[final_hypothesis[d][e]] += pTemp->classifier->composite_weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
							argmax[final_hypothesis[d][e]] += pTemp->classifier->composite_weightN * dfinal_hypothesis[d][e];
							pTemp = pTemp->next;
						}
					}

					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> majority voting + aktivacia
		case 13:
		case 23:
			{
			strcat(file_name, "log-test-13_23");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0;
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
						pTemp = pTemp->next;
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					
					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> aktivacia bitov + Htfieldd
		case 30:
			{
			strcat(file_name, "log-test-30");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}
			
			
			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight / sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight / sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				dataset[i]->setup_Htfield(rbfPAT->size_x, rbfPAT->size_y);
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					double ***binary_output;
					binary_output = new double**[learnt_datasets];
					for (int d = 0 ; d < learnt_datasets ; d++) {
						binary_output[d] = new double*[dataset_classifiers[d]];
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							binary_output[d][e] = new double[rbfPAT->size_x*rbfPAT->size_y];
							for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
								binary_output[d][e][s] = 0.0;
							}
						}
					}
					
					double ***binary_composite_output;
					binary_composite_output = new double**[learnt_datasets];
					for (int d = 0 ; d < learnt_datasets ; d++) {
						binary_composite_output[d] = new double*[dataset_classifiers[d]];
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							binary_composite_output[d][e] = new double[rbfPAT->size_x*rbfPAT->size_y];
							for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
								binary_composite_output[d][e][s] = 0.0;
							}
						}
					}

					dataset[i]->pattern[j].Htfield = new int*[rbfPAT->size_x];
					for (int m = 0 ; m < rbfPAT->size_x ; m++) {
						dataset[i]->pattern[j].Htfield[m] = new int[rbfPAT->size_y];
						for (int n = 0 ; n < rbfPAT->size_y ; n++) {
							dataset[i]->pattern[j].Htfield[m][n] = 0;
						}
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						//ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {			
							pTemp->classifier->assign_input(dataset[i], j);
							pTemp->classifier->feed_forward(); 
							pTemp->classifier->get_output(alg_radius);
							for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
								for (int s = 0 ; s < rbfPAT->size_x*rbfPAT->size_y ; s++) {
									binary_output[d][e][s] = pTemp->classifier->get_bit_output(s);
								}
								
							}
							pTemp = pTemp->next;
						}
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					for (int b = 0 ; b < rbfPAT->size_x*rbfPAT->size_y ; b++) {
						
						pTemp = pHead;
						for (int d = 0 ; d < learnt_datasets ; d++) {
							//ensemble *pD = pTemp;
							for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
								
								double value = 0.0;

								//ensemble *pE = pD;
								for (int s = 0 ; s <= e ; s++) {
									value += binary_output[d][s][b];
									
									//pE = pE->next;
								}
								//free(pE);
								value /= e + 1;

								//bity
								binary_composite_output[d][e][b] = value;// * pTemp->classifier->composite_weightN;
																
								pTemp = pTemp->next;
								
							}
							//free(pD);
						}
					}

					int tmp = 0;
					for (int m = 0 ; m < rbfPAT->size_x ; m++) {
						for (int n = 0 ; n < rbfPAT->size_y ; n++) {
							double value = 0.0;
							
							for (int d = 0 ; d < learnt_datasets ; d++) {
								for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
									value += binary_composite_output[d][e][tmp];
								}
							}
							tmp++;
							value /= current_ensemble;

							if (value > 0.0) {
								dataset[i]->pattern[j].Htfield[m][n] = 1;
							}
							if (value < 0.0) {
								dataset[i]->pattern[j].Htfield[m][n] = -1;
							}
							
							dataset[i]->pattern[j].Htfieldd[m][n] = value;
						}
					}

					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							delete[] binary_output[d][e];
							
						}
						delete [] binary_output[d];
					}
					delete [] binary_output;
					
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							delete[] binary_composite_output[d][e];
							
						}
						delete [] binary_composite_output[d];
					}
					delete [] binary_composite_output;

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> majority voting on RBF - obycajne
		case 35:
		case 36:
			{
			strcat(file_name, "log-test-35_36");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
						argclass[a] = 0;
					
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						pTemp->classifier->get_confidence(alg_radius);
						if (alg_voting_scheme == 35) {
							argmax[pTemp->classifier->temporal_class] += pTemp->classifier->temporal_confidence;
							
						}
						else {
							//kompletna aktivacia na vystupe
							for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
								int membership = pTemp->classifier->p_OUT->plane[p].membership;
								float confidence = pTemp->classifier->p_OUT->plane[p].conf;
								argmax[membership] += confidence;
								
							}
						}

						pTemp = pTemp->next;
					}
					
					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
		//Learn++ -> product voting - kompozitne
		case 37:
		case 38:
			{
			strcat(file_name, "log-test-37_38");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						pTemp->classifier->get_confidence(alg_radius);
						pTemp = pTemp->next;
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								if (alg_voting_scheme == 37) {
									argmax[pE->classifier->temporal_class] +=  pE->classifier->temporal_confidence;
									
								}
								else {
									//kompletna aktivacia na vystupe
									for (int p = 0 ; p < pE->classifier->p_OUT->this_planes ; p++) {
										int membership = pE->classifier->p_OUT->plane[p].membership;
										float confidence = pE->classifier->p_OUT->plane[p].conf;
										argmax[membership] += confidence;
										
									}
								}
								pE = pE->next;
							}
							//free(pE);
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}
							
							dfinal_hypothesis[d][e] = max;
							final_hypothesis[d][e] = winner;
							//final_hypothesis[d][e] = pTemp->classifier->temporal_class;
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}
					
					//finalna hypoteza po skupinach
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							if (argmax[final_hypothesis[d][e]] == 0.0) {
								//argmax[final_hypothesis[d][e]] += ((double)1.0 / pTemp->classifier->temporal_activation);
								argmax[final_hypothesis[d][e]] += dfinal_hypothesis[d][e];
							}
							else {
								//argmax[final_hypothesis[d][e]] *= ((double)1.0 / pTemp->classifier->temporal_activation);
								argmax[final_hypothesis[d][e]] += dfinal_hypothesis[d][e];
								//argmax[final_hypothesis[d][e]] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
							}
							pTemp = pTemp->next;
						}
					}

					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;

		//Learn++ -> product voting - obycajne
		case 40:
		case 41:
			{
			strcat(file_name, "log-test-40_41");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
						argclass[a] = 0;
					
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						
						if (alg_voting_scheme == 40) {
							if (argmax[pTemp->classifier->temporal_class] == 0.0) {
								argmax[pTemp->classifier->temporal_class] += ((double)1.0 / pTemp->classifier->temporal_activation);
							}
							else {
								argmax[pTemp->classifier->temporal_class] *= ((double)1.0 / pTemp->classifier->temporal_activation);
							}
						}
						else {
							//kompletna aktivacia na vystupe
							for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
								int membership = pTemp->classifier->p_OUT->plane[p].membership;
								float activation = pTemp->classifier->p_OUT->plane[p].x;
								if (argmax[membership] == 0.0) {
									argmax[membership] += ((double)1.0 / activation);
								}
								else {
									argmax[membership] *= ((double)1.0 / activation);
								}
							}
						}

						pTemp = pTemp->next;
					}
					
					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;

		//Learn++ -> product voting - kompozitne
		case 42:
		case 43:
			{
			strcat(file_name, "log-test-42_43");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
			
			fprintf(log_test, "\nDataset\tPattern");
			for (int a = 0 ; a < alg_classes ; a++) {
				fprintf(log_test, "\t%d", class_pointer[a].name);
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
		
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}
			
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {				
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
										
						pTemp = pTemp->next;
					}
					
					//potrebujeme zratat vystupy casti ensemblov pre jednotlive datasety (akokeby zdruzene ensemble)
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								argmax[a] = 0.0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								if (alg_voting_scheme == 42) {
									if (argmax[pE->classifier->temporal_class] == 0.0) {
										argmax[pE->classifier->temporal_class] += ((double)1.0 / pE->classifier->temporal_activation);
									}
									else {
										argmax[pE->classifier->temporal_class] *= ((double)1.0 / pE->classifier->temporal_activation);
									}
								}
								else {
									//kompletna aktivacia na vystupe
									for (int p = 0 ; p < pE->classifier->p_OUT->this_planes ; p++) {
										int membership = pE->classifier->p_OUT->plane[p].membership;
										float activation = pE->classifier->p_OUT->plane[p].x;
										if (argmax[membership] == 0.0) {
											argmax[membership] += ((double)1.0 / activation);
										}
										else {
											argmax[membership] *= ((double)1.0 / activation);
										}
									}
								}
								pE = pE->next;
							}
							//free(pE);
							
							int winner = 0;
							float max = argmax[0];	
							// maximum ak mame klasicke neurony na vystupe
							for (int a = 1 ; a < alg_classes ; a++) {
								if (argmax[a] > max) {
									max = argmax[a];
									winner = a;
								}
							}
							
							dfinal_hypothesis[d][e] = max;
							final_hypothesis[d][e] = winner;
							//final_hypothesis[d][e] = pTemp->classifier->temporal_class;
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}
					
					//finalna hypoteza po skupinach
					for (int a = 0 ; a < alg_classes ; a++) {
						argmax[a] = 0.0;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							if (argmax[final_hypothesis[d][e]] == 0.0) {
								//argmax[final_hypothesis[d][e]] += ((double)1.0 / pTemp->classifier->temporal_activation);
								argmax[final_hypothesis[d][e]] += dfinal_hypothesis[d][e];
							}
							else {
								//argmax[final_hypothesis[d][e]] *= ((double)1.0 / pTemp->classifier->temporal_activation);
								argmax[final_hypothesis[d][e]] += dfinal_hypothesis[d][e];
								//argmax[final_hypothesis[d][e]] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
							}
							pTemp = pTemp->next;
						}
					}

					int winner = 0;
					float max = argmax[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (argmax[a] > max) {
							max = argmax[a];
							winner = a;
						}
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = winner;
					
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", argmax[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
	//Learn++ -> pseudoloss cez hypotezy  
		case 50:
			{
			strcat(file_name, "log-test-50");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}

			//double *argmax = new double[alg_classes];

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}

			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						
						argmax[a] = 0;
					}
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						pTemp->classifier->get_confidence(alg_radius);
						
						double *prnt = new double[alg_classes];
						double *act = new double[alg_classes];
						for (int p = 0 ; p < alg_classes ; p++) {
							prnt[p] = 0.0;
							act[p] = 0.0;
						}
						//kompletna aktivacia na vystupe
						for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
							int membership = pTemp->classifier->p_OUT->plane[p].membership;
							float confidence = pTemp->classifier->p_OUT->plane[p].conf;
							prnt[membership] = confidence;
							act[membership] = pTemp->classifier->p_OUT->plane[p].x;
							dataset[i]->pattern[j].Hconf[membership] += pTemp->classifier->weightN * confidence;
						}
						
						delete[] prnt;
						delete[] act;
						pTemp = pTemp->next;
					}
					
					//aplikacia SOFTMAX na ziskanie confidence
					double confidence_sum = 0.0;
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
						dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
						confidence_sum += dataset[i]->pattern[j].Hconf[a];
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] /= confidence_sum;
						confidence[a][total] = dataset[i]->pattern[j].Hconf[a] / confidence_sum;
					}
					
					//docasne ulozenie Ht pre dany priklad					
					dataset[i]->pattern[j].Ht = 0;
					float max = dataset[i]->pattern[j].Hconf[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (dataset[i]->pattern[j].Hconf[a] > max) {
							max = dataset[i]->pattern[j].Hconf[a];
							dataset[i]->pattern[j].Ht = a;
						}
					}
				
					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", dataset[i]->pattern[j].Hconf[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			//delete[] argmax;
			}
			break;
		//Learn++ -> pseudoloss cez kompozitne hypotezy  
		case 51:
			{
			strcat(file_name, "log-test-51");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}

			//double *argmax = new double[alg_classes];
			double ***composite = new double**[learnt_datasets];
			for (int d = 0 ; d < learnt_datasets ; d++) {
				composite[d] = new double*[dataset_classifiers[d]];
				for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
					composite[d][e] = new double[alg_classes];
				}
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}

			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						
						argmax[a] = 0;
					}
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						pTemp->classifier->get_confidence(alg_radius);
						
						pTemp = pTemp->next;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp;
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							
							for (int a = 0 ; a < alg_classes ; a++) {
								composite[d][e][a] = 0.0;
							}

							ensemble *pE = pD;
							for (int s = 0 ; s <= e ; s++) {
								
								//kompletna aktivacia na vystupe
								for (int p = 0 ; p < pE->classifier->p_OUT->this_planes ; p++) {
									int membership = pE->classifier->p_OUT->plane[p].membership;
									float confidence = pE->classifier->p_OUT->plane[p].conf;
									composite[d][e][membership] += pE->classifier->weightN * confidence;
								}
								//argmax[pE->classifier->temporal_class] += pE->classifier->weightN * ((double)1.0 / pE->classifier->temporal_activation);
								pE = pE->next;
							}
							//free(pE);
							
							pTemp = pTemp->next;
						}
						//free(pD);
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						ensemble *pD = pTemp; 
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
							for (int a = 0 ; a < alg_classes ; a++) {
								dataset[i]->pattern[j].Hconf[a] +=  pD->classifier->composite_weightN * composite[d][e][a];
							}
							pD = pD->next;
						}
						pTemp = pTemp->next;
					}


					//aplikacia SOFTMAX na ziskanie confidence
					double confidence_sum = 0.0;
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
						dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
						confidence_sum += dataset[i]->pattern[j].Hconf[a];
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] /= confidence_sum;
						confidence[a][total] = dataset[i]->pattern[j].Hconf[a] / confidence_sum;
					}
					
					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = 0;
					float max = dataset[i]->pattern[j].Hconf[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (dataset[i]->pattern[j].Hconf[a] > max) {
							max = dataset[i]->pattern[j].Hconf[a];
							dataset[i]->pattern[j].Ht = a;
						}
					}

					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", dataset[i]->pattern[j].Hconf[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}

			for (int d = 0 ; d < learnt_datasets ; d++) {
				for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
					delete[] composite[d][e];
				}
				delete[] composite[d];
			}
			delete[] composite;

			}
			break;
			
		//Learn++ -> pseudoloss cez kompozitne hypotezy - po jednej kompozitnej hyp cez skupinu  
		case 52:
			{
			strcat(file_name, "log-test-52");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}

			//double *argmax = new double[alg_classes];
			double **composite = new double*[learnt_datasets];
			for (int d = 0 ; d < learnt_datasets ; d++) {
				composite[d] = new double[alg_classes];
			}

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}

			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						
						argmax[a] = 0;
					}
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}
					
					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						pTemp->classifier->assign_input(dataset[i], j);
						pTemp->classifier->feed_forward(); 
						pTemp->classifier->get_output(alg_radius);
						pTemp->classifier->get_confidence(alg_radius);
						
						pTemp = pTemp->next;
					}
					
					pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						
						for (int a = 0 ; a < alg_classes ; a++) {
							composite[d][a] = 0.0;
						}
						
						for (int e = 0 ; e < dataset_classifiers[d] ; e++) {
								
							//kompletna aktivacia na vystupe
							for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
								int membership = pTemp->classifier->p_OUT->plane[p].membership;
								float confidence = pTemp->classifier->p_OUT->plane[p].conf;
								composite[d][membership] += pTemp->classifier->weightN * confidence;
							}
							//argmax[pTemp->classifier->temporal_class] += pTemp->classifier->weightN * ((double)1.0 / pTemp->classifier->temporal_activation);
														
							pTemp = pTemp->next;
						}
					}
					
					//pTemp = pHead;
					for (int d = 0 ; d < learnt_datasets ; d++) {
						//ensemble *pD = pTemp; 
						for (int a = 0 ; a < alg_classes ; a++) {
							dataset[i]->pattern[j].Hconf[a] += /*pD->classifier->composite_weightN */ composite[d][a];
						}
						
						//pTemp = pTemp->next;
					}


					//aplikacia SOFTMAX na ziskanie confidence
					double confidence_sum = 0.0;
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
						dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
						confidence_sum += dataset[i]->pattern[j].Hconf[a];
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
						//dataset[i]->pattern[j].Hconf[a] /= confidence_sum;
						confidence[a][total] = dataset[i]->pattern[j].Hconf[a] / confidence_sum;
					}
					
					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = 0;
					float max = dataset[i]->pattern[j].Hconf[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (dataset[i]->pattern[j].Hconf[a] > max) {
							max = dataset[i]->pattern[j].Hconf[a];
							dataset[i]->pattern[j].Ht = a;
						}
					}

					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", dataset[i]->pattern[j].Hconf[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}

			for (int d = 0 ; d < learnt_datasets ; d++) {
				delete[] composite[d];
			}
			delete[] composite;
			}
			break;

		//Learn++ -> pseudoloss kompozitna hypoteza  
		case 60:
			{
			strcat(file_name, "log-test-60");	
			if( !(log_test = fopen(file_name,"wt"))) {
				printf("\nError in creating test file !");
				fprintf(log_screen, "\nError in creating test file !");
				exit (0);
			}
				
			//double *argmax = new double[alg_classes];

			//mormalizacia vah klasifikatorov 
			//spocitaju sa vsetky vahy
			double sum_of_weight = 0.0;
			double sum_of_composite_weight = 0.0;
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				sum_of_weight += pTemp->classifier->weight;
				sum_of_composite_weight += pTemp->classifier->composite_weight;
				pTemp = pTemp->next;
			}
			
			//prenastavenie vah
			pTemp = pHead;
			for (int t = 0 ; t < current_ensemble ; t++) {
				pTemp->classifier->weightN = pTemp->classifier->weight ;// sum_of_weight;
				pTemp->classifier->composite_weightN = pTemp->classifier->composite_weight ;// sum_of_composite_weight;
				pTemp = pTemp->next;
			}
			
			for (int a = 0 ; a < alg_classes ; a++) {
						
				//cez vsetky klasifikatory, ktore vedia klasifikovat triedu i 
				pTemp = pHead;
				for (int t = 0 ; t < current_ensemble ; t++) {
					if (true == search_list(class_pointer[a].name , pTemp->classifier->classes, pTemp->classifier->number_of_classes)) {
						class_pointer[a].flag++;
					}
					pTemp = pTemp->next;
				}
			}

			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					//reset pomocnych premennych
					for (int a = 0 ; a < alg_classes ; a++) {
						
						argmax[a] = 0;
					}
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}

					//cez vsetky klasifikatory
					pTemp = pHead;
					for (int t = 0 ; t < current_ensemble ; t++) {
						
						//if (pTemp->classifier->dataset == i) {
							pTemp->classifier->assign_input(dataset[i], j);
							pTemp->classifier->feed_forward(); 
							pTemp->classifier->get_output(alg_radius);
							pTemp->classifier->get_confidence(alg_radius);
							
							double *prnt = new double[alg_classes];
							double *act = new double[alg_classes];
							for (int p = 0 ; p < alg_classes ; p++) {
								prnt[p] = 0.0;
								act[p] = 0.0;
							}
							//kompletna aktivacia na vystupe
							for (int p = 0 ; p < pTemp->classifier->p_OUT->this_planes ; p++) {
								int membership = pTemp->classifier->p_OUT->plane[p].membership;
								float confidence = pTemp->classifier->p_OUT->plane[p].conf;
								prnt[membership] = confidence;
								act[membership] = pTemp->classifier->p_OUT->plane[p].x;
								dataset[i]->pattern[j].Hconf[membership] += pTemp->classifier->weightN * confidence;
							}
							
							delete[] prnt;
							delete[] act;
						//}

						pTemp = pTemp->next;
					}
					
					//aplikacia SOFTMAX na ziskanie confidence
					double sum = 0.0;
					for (int a = 0 ; a < alg_classes ; a++) {
						if (dataset[i]->pattern[j].Hconf[a] != 0.0) {
							dataset[i]->pattern[j].Hconf[a] = exp(dataset[i]->pattern[j].Hconf[a]);
						}
						//dataset[i]->pattern[j].Hconf[a] = dataset[i]->pattern[j].Hconf[a];
						sum += dataset[i]->pattern[j].Hconf[a];
					}
					
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] /= sum;
					}

					//docasne ulozenie Ht pre dany priklad
					dataset[i]->pattern[j].Ht = 0;
					float max = dataset[i]->pattern[j].Hconf[0];	
					// maximum ak mame klasicke neurony na vystupe
					for (int a = 1 ; a < alg_classes ; a++) {
						if (dataset[i]->pattern[j].Hconf[a] > max) {
							max = dataset[i]->pattern[j].Hconf[a];
							dataset[i]->pattern[j].Ht = a;
						}
					}

					fprintf(log_test, "\n%d\t%d", i, j);
					for (int a = 0 ; a < alg_classes ; a++) {
						fprintf(log_test, "\t%f", dataset[i]->pattern[j].Hconf[a]);
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
			}
			}
			break;
	}
	
	fclose(log_test);

	strcpy(file_name, working_directory);
	strcat(file_name, "report");

	if( !(log_report = fopen(file_name,"wt"))) {
		printf("\nError in creating report file !");
		fprintf(log_screen, "\nError in creating report file !");
		exit (0);
	}
	else {
		double *output;
		output = new double[rbfPAT->number_of_pat];
		
		fprintf(log_screen, "\nReport Saved !");

		fprintf(log_report, "Pattern             \tClass\tResult");
		fprintf(log_report, "\n-----------------------------------------");
		total = 0;
		for (int i = 0 ; i < alg_datasets ; i++) {
			dataset_counter[i] = 0;
			dataset_accuracy[i] = 0.0;
			for (int j = 0 ; j < dataset[i]->patterns ; j++) {
				
				if (alg_voting_scheme == 30) {
																	
					for (int s = 0 ; s < rbfPAT->number_of_pat ; s++) {
						output[s] = 0.0;
						for (int m = 0 ; m < rbfPAT->size_x ; m++) {
							for (int n = 0 ; n < rbfPAT->size_y ; n++) {
								//vzialenost vystupu od patternu
								if (alg_voting_scheme == 30) {
									output[s] += (dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfieldd[m][n] - rbfPAT->pattern[s].pixel[m][n]);
								}
								else {
									output[s] += (dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n])*(dataset[i]->pattern[j].Htfield[m][n] - rbfPAT->pattern[s].pixel[m][n]);
								}
							}
						}
					}
					
					double min = output[0];
					int choosen = 0;
					confidence[0][total] = output[0];
					for (int s = 1 ; s < rbfPAT->number_of_pat ; s++) {
						confidence[s][total] = output[s];
						if ( min > output[s]) {
							choosen = s;
							min = output[s];
						}
					}
					
					if (dataset[i]->pattern[j].membership == choosen) {
						total_counter++;
						dataset_counter[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t", i, j, dataset[i]->pattern[j].membership, choosen);
					}
					else {
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t*", i, j, dataset[i]->pattern[j].membership, choosen);
					}
					
					compare[choosen][dataset[i]->pattern[j].membership]++;
				}
				else {


					if (dataset[i]->pattern[j].Ht == dataset[i]->pattern[j].membership) {
						total_counter++;
						dataset_counter[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					else {
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t*", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					
					compare[dataset[i]->pattern[j].Ht][dataset[i]->pattern[j].membership]++;

				}

				//zapis konfidencnych hodnot
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", confidence[a][total]);
				}
				total++;
			}
			dataset_accuracy[i] = 100.0 * (float)dataset_counter[i] / (float)dataset[i]->patterns;
		}
		total_accuracy = 100.0 * (float)total_counter / (float)total;

		fprintf(log_report, "\n");
		fprintf(log_report, "\nDataset\tSamples\tCorrect\tWrong\tAccuracy");
		for (int i = 0 ; i < alg_datasets ; i++) {
			fprintf(log_report, "\n%d\t%d\t%d\t%d\t%f", i, dataset[i]->patterns, dataset_counter[i], dataset[i]->patterns-dataset_counter[i], dataset_accuracy[i]);
		}
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\nSum\t%d\t%d\t%d\t%f", total, total_counter, total-total_counter, total_accuracy);
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\n");
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\t%d", i);
		}
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\n%d", i);
			for (int j = 0 ; j < alg_classes ; j++) {
				fprintf(log_report, "\t%d", compare[i][j]);
			}
		}
		
		fclose(log_report);	
		fflush(log_screen);
		delete output;
	}

	delete composite_hypothesis;
	delete composite_hypothesis_value;
	
	delete argmax;
	delete argclass;
	
	delete[] dataset_classifiers;
	
	for (int i = 0 ; i < learnt_datasets ; i++) {
		delete[] final_hypothesis[i];
		delete[] dfinal_hypothesis[i];
	}
	delete[] final_hypothesis;
	delete[] dfinal_hypothesis;

	for (int i = 0 ; i < alg_classes ; i++) {
		delete[] confidence[i];
		delete[] compare[i];
	}
	delete[] confidence;
	delete[] compare;
}

//*********************************
//testovanie silnych klasifikatorov
void strong_testing_procedure(void) {
	
	int total = 0, total_counter = 0, total_rejected = 0, total_wrong = 0, total_ok = 0, *dataset_counter = new int[alg_datasets], *dataset_rejected = new int[alg_datasets],  *dataset_ok = new int[alg_datasets], *dataset_wrong = new int[alg_datasets];
	float total_accuracy = 0.0, total_rejrate = 0.0, total_reliability = 0.0, *dataset_accuracy = new float[alg_datasets], *dataset_rejrate = new float[alg_datasets], *dataset_reliability = new float[alg_datasets];
	char file_name[255];
	
	long total_samples = 0;
	for (int i = 0 ; i < alg_datasets ; i++) {
		total_samples += dataset[i]->patterns;
	}
	
	//tabulka vystupnych hodnot
	double **outputs = new double*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		outputs[i] = new double[total_samples];
		for (int j = 0 ; j < total_samples ; j++) {
			outputs[i][j] = 0.0;
		}
	}

	//konfidencna tabulka pre priklady
	double **confidence = new double*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		confidence[i] = new double[total_samples];
		for (int j = 0 ; j < total_samples ; j++) {
			confidence[i][j] = 0.0;
		}
	}

	//komparacna tabluka alg_classes*alg_classes
	int **compare = new int*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		compare[i] = new int[alg_classes];
		for (int j = 0 ; j < alg_classes ; j++) {
			compare[i][j] = 0;
		}
	}

	strcpy(file_name, working_directory);
	strcat(file_name, "report-strong");

	printf("\n\n >Testing strong classifiers");
	
	GetConsoleScreenBufferInfo(hStdout, &csbiInfo); 
	coordScreen.X = csbiInfo.dwCursorPosition.X;
	coordScreen.Y = csbiInfo.dwCursorPosition.Y;
	
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			sTemp = sHead;
			for (int i = 0 ; i < alg_datasets ; i++) {
				
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}

					sTemp->classifier->assign_input(dataset[i], j);
					sTemp->classifier->feed_forward(); 
					sTemp->classifier->get_output(alg_radius);
					sTemp->classifier->get_confidence(alg_radius);

					dataset[i]->pattern[j].Ht = sTemp->classifier->temporal_class;
					for (int a = 0 ; a < sTemp->classifier->number_of_classes ; a++) {
						int membership = sTemp->classifier->p_OUT->plane[a].membership;
						outputs[membership][total] = sTemp->classifier->p_OUT->plane[a].x;
						confidence[membership][total] = sTemp->classifier->p_OUT->plane[a].conf;
						dataset[i]->pattern[j].Hconf[membership] = sTemp->classifier->p_OUT->plane[a].conf;
					}

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
				sTemp = sTemp->next;
			}


	if( !(log_report = fopen(file_name,"wt"))) {
		printf("\nError in creating report file !");
		fprintf(log_screen, "\nError in creating report file !");
		exit (0);
	}
	else {
		
		fprintf(log_screen, "\nReport Saved !");

		fprintf(log_report, "Pattern             \tClass\tResult");
		fprintf(log_report, "\n-----------------------------------------");
		total = 0;
		for (int i = 0 ; i < alg_datasets ; i++) {
			dataset_counter[i] = 0;
			dataset_rejected[i] = 0;
			dataset_ok[i] = 0;
			dataset_wrong[i] = 0;
			dataset_accuracy[i] = 0.0;
			for (int j = 0 ; j < dataset[i]->patterns ; j++) {
				
				if (dataset[i]->pattern[j].Hconf[dataset[i]->pattern[j].Ht] >= alg_rejection_treshold) {

					if (dataset[i]->pattern[j].Ht == dataset[i]->pattern[j].membership) {
						total_ok++;
						dataset_ok[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					else {
						total_wrong++;
						dataset_wrong[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t*", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					
					total_counter++;
					dataset_counter[i]++;
					compare[dataset[i]->pattern[j].Ht][dataset[i]->pattern[j].membership]++;
				}
				else {
					total_rejected++;
					dataset_rejected[i]++;
					fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t+", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
				}

				//zapis vystupnych hodnot
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", outputs[a][total]);
				}
				//zapis konfidencnych hodnot
				fprintf(log_report, "\n\t\t\t\t\t");
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", confidence[a][total]);
				}
				
				total++;
			}
			dataset_accuracy[i] = 100.0 * (float)dataset_ok[i] / (float)dataset[i]->patterns;
			dataset_rejrate[i] = 100.0 * (float)dataset_rejected[i] / (float)dataset[i]->patterns;
			dataset_reliability[i] = 100.0 *dataset_accuracy[i] / ((float)100.0 - dataset_rejrate[i]);
		}
		total_accuracy = 100.0 * (float)total_ok / (float)total;
		total_rejrate = 100.0 * (float)total_rejected / (float)total;
		total_reliability = 100.0 * total_accuracy / ((float)100.0 - total_rejrate);

		fprintf(log_report, "\n");
		fprintf(log_report, "\nRadius = %f", alg_radius);
		fprintf(log_report, "\nRejection treshold = %f", alg_rejection_treshold);
		fprintf(log_report, "\n");
		fprintf(log_report, "\nDataset\tSamples\tCorrect\tWrong\tReject\tAccur\t\tRejRate\t\tReliability");
		for (int i = 0 ; i < alg_datasets ; i++) {
			fprintf(log_report, "\n%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f", i, dataset[i]->patterns, dataset_ok[i], dataset_wrong[i], dataset_rejected[i], dataset_accuracy[i], dataset_rejrate[i], dataset_reliability[i]);
		}
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\nSum\t%d\t%d\t%d\t%d\t%f\t%f\t%f", total, total_ok, total_wrong, total_rejected, total_accuracy, total_rejrate, total_reliability);
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\n");
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\t%d", i);
		}
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\n%d", i);
			for (int j = 0 ; j < alg_classes ; j++) {
				fprintf(log_report, "\t%d", compare[i][j]);
			}
		}
		
		fclose(log_report);	
		fflush(log_screen);

	}
	
	for (int i = 0 ; i < alg_classes ; i++) {
		delete[] outputs[i];
		delete[] confidence[i];
		delete[] compare[i];
	}
	delete[] outputs;
	delete[] confidence;
	delete[] compare;

}

//****************************************************************
//testovanie mikrodkopickych snimok chromozomov skorekcnym modulom
void pap_testing_procedure(void) {
	
	//statistika na mnozine (mnozinach) prikladov
	//zatial staticky
	int stat_min[25];
	int stat_max[25];
	int stat_avg[25];
	int stat_cou[25];

	//nacitanie zo suboru
	FILE *fl;
	if ((fl = fopen(alg_lenghts_file,"rt")) == NULL) {
		printf("\nError in opening lenghts file !!!");
	}
	else {
		char chtmp[255];
		int itmp;
		fscanf(fl, "%s", &chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		for (int i = 0 ; i < 25 ; i++) {
			fscanf(fl, "%d", &itmp);
			fscanf(fl, "%d", &stat_cou[i]);
			fscanf(fl, "%d", &stat_min[i]);
			fscanf(fl, "%d", &stat_max[i]);
			fscanf(fl, "%d", &stat_avg[i]);
		}
	}
	
	int total = 0, total_counter = 0, total_rejected = 0, total_wrong = 0, total_ok = 0, *dataset_counter = new int[alg_datasets], *dataset_rejected = new int[alg_datasets],  *dataset_ok = new int[alg_datasets], *dataset_wrong = new int[alg_datasets];
	float total_accuracy = 0.0, total_rejrate = 0.0, total_reliability = 0.0, *dataset_accuracy = new float[alg_datasets], *dataset_rejrate = new float[alg_datasets], *dataset_reliability = new float[alg_datasets];
	char file_name[255];
	
	long total_samples = 0;
	for (int i = 0 ; i < alg_datasets ; i++) {
		total_samples += dataset[i]->patterns;
	}
	
	double ***outputs = new double**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		outputs[i] = new double*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			outputs[i][j] = new double[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				outputs[i][j][k] = 0.0;
			}
		}
	}

	double ***confidence = new double**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		confidence[i] = new double*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			confidence[i][j] = new double[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				confidence[i][j][k] = 0.0;
			}
		}
	}

	int ***sample_sort = new int**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		sample_sort[i] = new int*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			sample_sort[i][j] = new int[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				sample_sort[i][j][k] = -1;
			}
		}
	}
	
	//poradie vramci PAP datasetu
	int **karyogram_sort = new int*[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		karyogram_sort[i] = new int[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			karyogram_sort[i][j] = -1;
		}
	}
	
	//karyogram
	int chromosomes[24];
	int chromosomeX;
	int chromosomeY;
	
	//komparacna tabluka alg_classes*alg_classes
	int **compare = new int*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		compare[i] = new int[alg_classes];
		for (int j = 0 ; j < alg_classes ; j++) {
			compare[i][j] = 0;
		}
	}

	strcpy(file_name, working_directory);
	strcat(file_name, "report-strong");

	printf("\n\n >Testing strong classifiers");
	
	GetConsoleScreenBufferInfo(hStdout, &csbiInfo); 
	coordScreen.X = csbiInfo.dwCursorPosition.X;
	coordScreen.Y = csbiInfo.dwCursorPosition.Y;
	
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			sTemp = sHead;
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}
					

					sTemp->classifier->assign_input(dataset[i], j);
					sTemp->classifier->feed_forward(); 
					sTemp->classifier->get_output(alg_radius);
					sTemp->classifier->get_confidence(alg_radius);

					//dataset[i]->pattern[j].Ht = sTemp->classifier->temporal_class;
					for (int a = 0 ; a < sTemp->classifier->number_of_classes ; a++) {
						int membership = sTemp->classifier->p_OUT->plane[a].membership;
						outputs[i][j][membership] = sTemp->classifier->p_OUT->plane[a].x;
						confidence[i][j][membership] = sTemp->classifier->p_OUT->plane[a].conf;
						dataset[i]->pattern[j].Hconf[membership] = sTemp->classifier->p_OUT->plane[a].conf;
					}

					//zoradenie podla konfidencie
					int counter = 0;
					for (int a = 0 ; a < alg_classes ; a++) {
						
						int max = 0;
						double max_conf = 0.0;
						for (int b = 0 ; b < alg_classes ; b++) {
							if (max_conf < confidence[i][j][b] && sample_sort[i][j][b] == -1) {
								max_conf = confidence[i][j][b];
								max = b;
							}
						}
						sample_sort[i][j][max] = counter;
						counter++;
					}
					
					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
				//sTemp = sTemp->next;
			}
			
			//budovanie karyogramov 
			for (int i = 0 ; i < alg_datasets ; i++) {
				
				//obsadenost karyogramu
				for (int a = 0 ; a < 24 ; a++) {
					chromosomes[a] = 0;
				}
				chromosomeX = 0; //23
				chromosomeY = 0; //24

				//zoradenie podla doveryhodnosti na vystupe
				int counter = 0;
				for (int a = 0 ; a < dataset[i]->patterns ; a++) {
						
					int max = 0;
					double max_conf = 0.0;
					for (int b = 0 ; b < dataset[i]->patterns ; b++) {
						int iter = 0;
						while (sample_sort[i][b][iter] != 0) {
							iter++;
						}
						double conf = confidence[i][b][iter];
						if (max_conf < conf && karyogram_sort[i][b] == -1) {
							max_conf = conf;
							max = b;
						}
					}
					karyogram_sort[i][max] = counter;
					counter++;
				}
			
				/*
				//bez korekcie
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					int sample = 0;
					while (karyogram_sort[i][sample] != j) {
						sample++;
					}
					int iter = 0;
					while (sample_sort[i][sample][iter] != 0) {
						iter++;
					}
					dataset[i]->pattern[sample].Ht = iter;
				}
				*/
				// s korekciou
				
				int order = 0;
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					int sample = 0;
					while (karyogram_sort[i][sample] != j) {
						sample++;
					}
					int iter = 0;
					while (sample_sort[i][sample][iter] != order) {
						iter++;
					}
					
					if (order > 24) {
						for (int k = j ; k < dataset[i]->patterns ; k++) {
							dataset[i]->pattern[k].Ht = 0;
						}
						break;
					}

					//if (chromosomes[iter] < 2) {
					
					bool length_sign = false;
					//group A+B
					if (iter >= 1 && iter <= 5) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 50 && dataset[i]->pattern[sample].length <= 100) {
						//edi
						if (dataset[i]->pattern[sample].length >= 60 && dataset[i]->pattern[sample].length <= 100) {
							length_sign = true;
						}
					}
					//group C+X
					if ((iter >= 6 && iter <= 12) || iter == 23) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 40 && dataset[i]->pattern[sample].length <= 90) {
						//edi
						if (dataset[i]->pattern[sample].length >= 40 && dataset[i]->pattern[sample].length <= 80) {
							length_sign = true;
						}
					}
					//group D
					if (iter >= 13 && iter <= 15) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 30 && dataset[i]->pattern[sample].length <= 60) {
						//edi
						if (dataset[i]->pattern[sample].length >= 33 && dataset[i]->pattern[sample].length <= 60) {
							length_sign = true;
						}
					}
					//group E
					if (iter >= 16 && iter <= 18) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 27 && dataset[i]->pattern[sample].length <= 50) {
						//edi
						if (dataset[i]->pattern[sample].length >= 27 && dataset[i]->pattern[sample].length <= 47) {
							length_sign = true;
						}
					}
					//group F
					if (iter >= 19 && iter <= 20) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 21 && dataset[i]->pattern[sample].length <= 50) {
						//edi
						if (dataset[i]->pattern[sample].length >= 15 && dataset[i]->pattern[sample].length <= 40) {
							length_sign = true;
						}
					}
					//group G+Y
					if ((iter >= 21 && iter <= 22) || iter == 24) {
						//cph
						//if (dataset[i]->pattern[sample].length >= 0 && dataset[i]->pattern[sample].length <= 40) {
						//edi
						if (dataset[i]->pattern[sample].length >= 0 && dataset[i]->pattern[sample].length <= 35) {
							length_sign = true;
						}
					}

					if (chromosomes[iter] < 2 && length_sign == true) {
					
						dataset[i]->pattern[sample].Ht = iter;
						chromosomes[iter]++;
						order = 0;
					}
					else {
						order++;
						if (order < alg_classes) {
							j--;
						}
						else {
							dataset[i]->pattern[sample].Ht = 0;
						}
					}
				}
				
			}

	if( !(log_report = fopen(file_name,"wt"))) {
		printf("\nError in creating report file !");
		fprintf(log_screen, "\nError in creating report file !");
		exit (0);
	}
	else {
		
		fprintf(log_screen, "\nReport Saved !");

		fprintf(log_report, "Pattern             \tClass\tResult");
		fprintf(log_report, "\n-----------------------------------------");
		total = 0;
		for (int i = 0 ; i < alg_datasets ; i++) {
			dataset_counter[i] = 0;
			dataset_rejected[i] = 0;
			dataset_ok[i] = 0;
			dataset_wrong[i] = 0;
			dataset_accuracy[i] = 0.0;
			for (int j = 0 ; j < dataset[i]->patterns ; j++) {
				
				if (dataset[i]->pattern[j].Ht <= 0 || dataset[i]->pattern[j].Ht >= 24) {
					dataset[i]->pattern[j].Ht = 0;
				}

				if (dataset[i]->pattern[j].Hconf[dataset[i]->pattern[j].Ht] >= alg_rejection_treshold) {

					if (dataset[i]->pattern[j].Ht == dataset[i]->pattern[j].membership) {
						total_ok++;
						dataset_ok[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					else {
						total_wrong++;
						dataset_wrong[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t*", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					
					total_counter++;
					dataset_counter[i]++;
					compare[dataset[i]->pattern[j].Ht][dataset[i]->pattern[j].membership]++;
				}
				else {
					total_rejected++;
					dataset_rejected[i]++;
					fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t+", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
				}

				//zapis vystupnych hodnot
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", outputs[i][j][a]);
				}
				//zapis konfidencnych hodnot
				fprintf(log_report, "\n\t\t\t\t\t");
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", confidence[i][j][a]);
				}
				
				total++;
			}
			dataset_accuracy[i] = 100.0 * (float)dataset_ok[i] / (float)dataset[i]->patterns;
			dataset_rejrate[i] = 100.0 * (float)dataset_rejected[i] / (float)dataset[i]->patterns;
			dataset_reliability[i] = 100.0 *dataset_accuracy[i] / ((float)100.0 - dataset_rejrate[i]);
		}
		total_accuracy = 100.0 * (float)total_ok / (float)total;
		total_rejrate = 100.0 * (float)total_rejected / (float)total;
		total_reliability = 100.0 * total_accuracy / ((float)100.0 - total_rejrate);

		fprintf(log_report, "\n");
		fprintf(log_report, "\nRadius = %f", alg_radius);
		fprintf(log_report, "\nRejection treshold = %f", alg_rejection_treshold);
		fprintf(log_report, "\n");
		fprintf(log_report, "\nDataset\tSamples\tCorrect\tWrong\tReject\tAccur\t\tRejRate\t\tReliability");
		for (int i = 0 ; i < alg_datasets ; i++) {
			fprintf(log_report, "\n%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f", i, dataset[i]->patterns, dataset_counter[i], dataset_wrong[i], dataset_rejected[i], dataset_accuracy[i], dataset_rejrate[i], dataset_reliability[i]);
		}
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\nSum\t%d\t%d\t%d\t%d\t%f\t%f\t%f", total, total_counter, total_wrong, total_rejected, total_accuracy, total_rejrate, total_reliability);
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\n");
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\t%d", i);
		}
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\n%d", i);
			for (int j = 0 ; j < alg_classes ; j++) {
				fprintf(log_report, "\t%d", compare[i][j]);
			}
		}
		
		fclose(log_report);	
		fflush(log_screen);

	}
	
	
	for (int i = 0 ; i < alg_datasets ; i++) {
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			delete[] outputs[i][j];
			delete[] confidence[i][j];
			delete[] sample_sort[i][j];
		}
		delete[] outputs[i];
		delete[] confidence[i]; 
		delete[] sample_sort[i];
	}
	delete[] outputs;
	delete[] confidence;
	delete[] sample_sort;
	
	delete[] dataset_counter;
	delete[] dataset_accuracy;

	for (int i = 0 ; i < alg_datasets ; i++) {
		delete[] karyogram_sort[i];
	}
	delete[] karyogram_sort;
	
	for (int i = 0 ; i < alg_classes ; i++) {
		delete[] compare[i];
	}
	delete[] compare;
}

//****************************************************************
//testovanie mikrodkopickych snimok chromozomov + priznak dlzky
void lenght_testing_procedure(void) {
	
	//statistika na mnozine (mnozinach) prikladov
	//zatial staticky
	int stat_min[25];
	int stat_max[25];
	int stat_avg[25];
	int stat_cou[25];

	//nacitanie zo suboru
	FILE *fl;
	if ((fl = fopen(alg_lenghts_file,"rt")) == NULL) {
		printf("\nError in opening lenghts file !!!");
	}
	else {
		char chtmp[255];
		int itmp;
		fscanf(fl, "%s", &chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		fscanf(fl, "%s",&chtmp);
		for (int i = 0 ; i < 25 ; i++) {
			fscanf(fl, "%d", &itmp);
			fscanf(fl, "%d", &stat_cou[i]);
			fscanf(fl, "%d", &stat_min[i]);
			fscanf(fl, "%d", &stat_max[i]);
			fscanf(fl, "%d", &stat_avg[i]);
		}
	}
	
	int total = 0, total_counter = 0, total_rejected = 0, total_wrong = 0, total_ok = 0, *dataset_counter = new int[alg_datasets], *dataset_rejected = new int[alg_datasets],  *dataset_ok = new int[alg_datasets], *dataset_wrong = new int[alg_datasets];
	float total_accuracy = 0.0, total_rejrate = 0.0, total_reliability = 0.0, *dataset_accuracy = new float[alg_datasets], *dataset_rejrate = new float[alg_datasets], *dataset_reliability = new float[alg_datasets];
	char file_name[255];
	
	long total_samples = 0;
	for (int i = 0 ; i < alg_datasets ; i++) {
		total_samples += dataset[i]->patterns;
	}
	
	double ***outputs = new double**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		outputs[i] = new double*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			outputs[i][j] = new double[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				outputs[i][j][k] = 0.0;
			}
		}
	}

	double ***confidence = new double**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		confidence[i] = new double*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			confidence[i][j] = new double[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				confidence[i][j][k] = 0.0;
			}
		}
	}

	int ***sample_sort = new int**[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		sample_sort[i] = new int*[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			sample_sort[i][j] = new int[alg_classes];
			for (int k = 0 ; k < alg_classes ; k++) {		
				sample_sort[i][j][k] = -1;
			}
		}
	}
	
	//poradie vramci PAP datasetu
	int **karyogram_sort = new int*[alg_datasets];
	for (int i = 0 ; i < alg_datasets ; i++) {
		karyogram_sort[i] = new int[dataset[i]->patterns];
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			karyogram_sort[i][j] = -1;
		}
	}
	
	//karyogram
	int chromosomes[24];
	int chromosomeX;
	int chromosomeY;
	
	//komparacna tabluka alg_classes*alg_classes
	int **compare = new int*[alg_classes];
	for (int i = 0 ; i < alg_classes ; i++) {
		compare[i] = new int[alg_classes];
		for (int j = 0 ; j < alg_classes ; j++) {
			compare[i][j] = 0;
		}
	}
	
	

	strcpy(file_name, working_directory);
	strcat(file_name, "report-strong");

	printf("\n\n >Testing strong classifiers");
	
	GetConsoleScreenBufferInfo(hStdout, &csbiInfo); 
	coordScreen.X = csbiInfo.dwCursorPosition.X;
	coordScreen.Y = csbiInfo.dwCursorPosition.Y;
	
			//testovacia procedura prejde vsetky priklady zo vsetkych datasetov
			sTemp = sHead;
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					
					
					if (dataset[i]->pattern[j].Hconf == NULL) {
						dataset[i]->pattern[j].Hconf = new double[alg_classes];
					}
					for (int a = 0 ; a < alg_classes ; a++) {
						dataset[i]->pattern[j].Hconf[a] = 0.0;
					}
					

					sTemp->classifier->assign_input(dataset[i], j);
					sTemp->classifier->feed_forward(); 
					sTemp->classifier->get_output(alg_radius);
					sTemp->classifier->get_confidence(alg_radius);

					//dataset[i]->pattern[j].Ht = sTemp->classifier->temporal_class;
					for (int a = 0 ; a < sTemp->classifier->number_of_classes ; a++) {
						int membership = sTemp->classifier->p_OUT->plane[a].membership;
						outputs[i][j][membership] = sTemp->classifier->p_OUT->plane[a].x;
						confidence[i][j][membership] = sTemp->classifier->p_OUT->plane[a].conf;
						dataset[i]->pattern[j].Hconf[membership] = sTemp->classifier->p_OUT->plane[a].conf;
					}

					//zoradenie podla konfidencie
					int counter = 0;
					for (int a = 0 ; a < alg_classes ; a++) {
						
						int max = 0;
						double max_conf = 0.0;
						for (int b = 0 ; b < alg_classes ; b++) {
							if (max_conf < confidence[i][j][b] && sample_sort[i][j][b] == -1) {
								max_conf = confidence[i][j][b];
								max = b;
							}
						}
						sample_sort[i][j][max] = counter;
						counter++;
					}
					
					/*
					//korekcny modul
					bool found = false;
					int found_counter = 0;
					while (found == false && found_counter < alg_classes) {
						//najdi maximalny confidence pre marker = 0
						int max = 0;
						double max_conf = confidence[0][total];
						for (int a = 1 ; a < alg_classes ; a++) {
							if (max_conf < confidence[a][total] && marker[a][total] == 0) {
								max_conf = confidence[a][total];
								max = a;
							}
						}
						//oznac priklad za uz vybraty
						marker[max][total] = 1;
						//skontroluj dlzku
						if (dataset[i]->pattern[j].length >= stat_min[max] && dataset[i]->pattern[j].length <= stat_max[max]) {
							dataset[i]->pattern[j].Ht = max;
							found = true;
						}
						found_counter++;
					}
					*/

					total++;
					SetConsoleCursorPosition( hStdout, coordScreen );
					printf("%d - dataset (%d) pattern (%d) ", total, i, j);
				}
				//sTemp = sTemp->next;
			}
			
			//korekcia konfidencie podla priemernych dlzok
			for (int i = 0 ; i < alg_datasets ; i++) {
				for (int j = 0 ; j < dataset[i]->patterns ; j++) {
					int lenght = dataset[i]->pattern[j].length;
					for (int k = 0 ; k < alg_classes ; k++) {
						
						int distance = abs(lenght - stat_avg[k]);
						confidence[i][j][k] *= (double)1.0 / ((double)distance + (double)20.0);
						
					}

					double max = 0.0;
					for (int k = 0 ; k < alg_classes ; k++) {
						if (max < confidence[i][j][k]) {
							max = confidence[i][j][k];
							dataset[i]->pattern[j].Ht = k;
						}
					}
				}
			}


	if( !(log_report = fopen(file_name,"wt"))) {
		printf("\nError in creating report file !");
		fprintf(log_screen, "\nError in creating report file !");
		exit (0);
	}
	else {
		
		fprintf(log_screen, "\nReport Saved !");

		fprintf(log_report, "Pattern             \tClass\tResult");
		fprintf(log_report, "\n-----------------------------------------");
		total = 0;
		for (int i = 0 ; i < alg_datasets ; i++) {
			dataset_counter[i] = 0;
			dataset_rejected[i] = 0;
			dataset_ok[i] = 0;
			dataset_wrong[i] = 0;
			dataset_accuracy[i] = 0.0;
			for (int j = 0 ; j < dataset[i]->patterns ; j++) {
				/*
				if (dataset[i]->pattern[j].Ht <= 0 || dataset[i]->pattern[j].Ht >= 25) {
					dataset[i]->pattern[j].Ht = 0;
				}
				*/
				if (dataset[i]->pattern[j].Hconf[dataset[i]->pattern[j].Ht] >= alg_rejection_treshold) {
				//if (confidence[i][j][dataset[i]->pattern[j].Ht] >= alg_rejection_treshold) {

					if (dataset[i]->pattern[j].Ht == dataset[i]->pattern[j].membership) {
						total_ok++;
						dataset_ok[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					else {
						total_wrong++;
						dataset_wrong[i]++;
						fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t*", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
					}
					
					total_counter++;
					dataset_counter[i]++;
					compare[dataset[i]->pattern[j].Ht][dataset[i]->pattern[j].membership]++;
				}
				else {
					total_rejected++;
					dataset_rejected[i]++;
					fprintf(log_report, "\nDataset(%d) Pat(%d)\t%d\t%d\t+", i, j, dataset[i]->pattern[j].membership, dataset[i]->pattern[j].Ht);
				}

				//zapis vystupnych hodnot
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", outputs[i][j][a]);
				}
				//zapis konfidencnych hodnot
				fprintf(log_report, "\n\t\t\t\t\t");
				for (int a = 0 ; a < alg_classes ; a++) {
					fprintf(log_report, "\t%f", confidence[i][j][a]);
				}
				
				total++;
			}
			dataset_accuracy[i] = 100.0 * (float)dataset_ok[i] / (float)dataset[i]->patterns;
			dataset_rejrate[i] = 100.0 * (float)dataset_rejected[i] / (float)dataset[i]->patterns;
			dataset_reliability[i] = 100.0 *dataset_accuracy[i] / ((float)100.0 - dataset_rejrate[i]);
		}
		total_accuracy = 100.0 * (float)total_ok / (float)total;
		total_rejrate = 100.0 * (float)total_rejected / (float)total;
		total_reliability = 100.0 * total_accuracy / ((float)100.0 - total_rejrate);

		fprintf(log_report, "\n");
		fprintf(log_report, "\nRadius = %f", alg_radius);
		fprintf(log_report, "\nRejection treshold = %f", alg_rejection_treshold);
		fprintf(log_report, "\n");
		fprintf(log_report, "\nDataset\tSamples\tCorrect\tWrong\tReject\tAccur\t\tRejRate\t\tReliability");
		for (int i = 0 ; i < alg_datasets ; i++) {
			fprintf(log_report, "\n%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f", i, dataset[i]->patterns, dataset_counter[i], dataset_wrong[i], dataset_rejected[i], dataset_accuracy[i], dataset_rejrate[i], dataset_reliability[i]);
		}
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\nSum\t%d\t%d\t%d\t%d\t%f\t%f\t%f", total, total_counter, total_wrong, total_rejected, total_accuracy, total_rejrate, total_reliability);
		fprintf(log_report, "\n-----------------------------------------");
		fprintf(log_report, "\n");
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\t%d", i);
		}
		for (int i = 0 ; i < alg_classes ; i++) {
			fprintf(log_report, "\n%d", i);
			for (int j = 0 ; j < alg_classes ; j++) {
				fprintf(log_report, "\t%d", compare[i][j]);
			}
		}
		
		fclose(log_report);	
		fflush(log_screen);

	}

	for (int i = 0 ; i < alg_datasets ; i++) {
		for (int j = 0 ; j < dataset[i]->patterns ; j++) {
			delete[] outputs[i][j];
			delete[] confidence[i][j];
			delete[] sample_sort[i][j];
		}
		delete[] outputs[i];
		delete[] confidence[i]; 
		delete[] sample_sort[i];
	}
	delete[] outputs;
	delete[] confidence;
	delete[] sample_sort;
	
	delete[] dataset_counter;
	delete[] dataset_accuracy;

	for (int i = 0 ; i < alg_datasets ; i++) {
		delete[] karyogram_sort[i];
	}
	delete[] karyogram_sort;
	
	for (int i = 0 ; i < alg_classes ; i++) {
		delete[] compare[i];
	}
	delete[] compare;
}

//***********
//ruleta SSwR
long SSwR (read_file *samples) {
	
	long individual = -1;
	//TRanrotBGenerator generator((unsigned)time(NULL));
	
	double select = generator.Random();
	double position = 0.0;
	//int a = 0;
	for (long i = 0 ; i < samples->patterns ; i++) {
		position += samples->pattern[i].weightN;
		//a = samples->pattern[i].membership;
		if (position > select) {
			individual = i;
			break;
		}
	}
	
	//moze sa stat ze ostane -1 (ak sa vyberie velmi blizka hodnota cislu 1)
	if (individual == -1) {
		//vyberiem posledneho
		individual = samples->patterns - 1;
	}

	return individual;
}

//************
//ruleta SSwoR
long SSwoR (read_file *samples) {

	long individual = -1;
	//TRanrotBGenerator generator((unsigned)time(NULL));
	
	double select = generator.Random();
	double position = 0.0;
	
	//skrat ruletu o doteraz vynechane priklady
	double reduction = 0.0;
	for (long i = 0 ; i < samples->patterns ; i++) {
		if (samples->pattern[i].repl_flag == 1) {
			reduction += samples->pattern[i].weightN;
		}
	}
	//nastavenie pomocnych vah
	reduction = (double)1.0 - reduction;
	for (long i = 0 ; i < samples->patterns ; i++) {
		samples->pattern[i].weightR = samples->pattern[i].weightN / reduction;
	}

	//zatocenie ruletou
	for (long i = 0 ; i < samples->patterns ; i++) {
		//vyberame len priklady ktore ostali na rulete
		if (samples->pattern[i].repl_flag == 0) {
			position += samples->pattern[i].weightR;
		}
		//a = samples->pattern[i].membership;
		if (position > select) {
			individual = i;
			samples->pattern[i].repl_flag = 1;
			break;
		}
	}
	
	//moze sa stat ze ostane -1 (ak sa vyberie velmi blizka hodnota cislu 1)
	if (individual == -1) {
		//vyberiem posledneho
		individual = samples->patterns - 1;
	}

	return individual;

}

//*********************
//otvori datovu mnozinu
read_file* open_dataset(char *name) {

	char name_img[255], name_lab[255], name_len[255];

	strcpy(name_img, name);
	strcat(name_img, ".images");
	strcpy(name_lab, name);
	strcat(name_lab, ".labels");
	strcpy(name_len, name);
	strcat(name_len, ".lenghts");

	read_file *tempset = new read_file();
	bool opened = tempset->open_file(name_img, name_lab, name_len);
		
	if (opened == true) {
		tempset->read_header();
		tempset->read_patterns();
		printf("\nReading dataset (%s) completed !", name);
		fprintf(log_screen, "\nReading dataset (%s) completed !", name);
		fflush(log_screen);
		return tempset;
	}
	else {
		printf("\nError in reading file !");
		fprintf(log_screen, "\nError in reading file !");
		fflush(log_screen);
		exit(0);
	}
}

//*********************
//zavrie datovu mnozinu
void close_dataset() {
	for (int i = 0 ; i < alg_datasets ; i++) {
		delete dataset[i];
	}
	delete dataset;
}

//*********************
//prehladavanie zoznamu
bool search_list(int value, int *list, int number) {
	
	for (int i = 0 ; i < number ; i++) {
		if (value == list[i]) {
				return true;
		}
	}
	return false;
}
bool search_long_list(long value, long *list, long number) {
	
	for (int i = 0 ; i < number ; i++) {
		if (value == list[i]) {
				return true;
		}
	}
	return false;
}

//**************************************
//vyber z mnoziny na zaklade distribucie 
long* pickup_samples(read_file *samples, int number_of_samples, int all) {
	
	long *samples_index = NULL;
	samples_index = new long[number_of_samples];
	
	if (all == 1) {
		for (int i = 0 ; i < number_of_samples ; i++) {
			//ulozenie indexov prikladov vyberu do pola
			samples_index[i] = i;	
		}
	}
	else {
		//SSwR
		for (int i = 0 ; i < number_of_samples ; i++) {
			//ulozenie indexov prikladov vyberu do pola
			samples_index[i] = SSwR(samples);	
		}
	}
	//SSwoR
	/*
	//nastavit priznaky vsetkych prikladov na 0
	for (int i = 0 ; i < samples->patterns ; i++) {
		samples->pattern[i].repl_flag = 0;
	}
	for (int i = 0 ; i < number_of_samples ; i++) {
		//ulozenie indexov prikladov vyberu do pola
		samples_index[i] = SSwoR(samples);	
	}
	*/
	return samples_index;
}

//*************************
//estimacia pred ucenim CNN 
void estimate_cnn(WeakCNN *net, read_file* samples, long* samples_index, int number_of_samples) {
	
	printf("\n\n-->Estimation : ");
	GetConsoleScreenBufferInfo(hStdout, &csbiInfo); 
	coordScreen.X = csbiInfo.dwCursorPosition.X;
	coordScreen.Y = csbiInfo.dwCursorPosition.Y;

	for (int h = 0 ; h < alg_estimation ; h++) {
		//nahodny vyber vzoru s mnoziny vzorov cez indexy
		long h_rand = (long)((rand()%(long)(number_of_samples)));
		
		SetConsoleCursorPosition( hStdout, coordScreen );
		printf("%d of %d",h+1, alg_estimation);

		net->assign_input(samples, samples_index[h_rand]);
		net->feed_forward();
		net->back_propagation_e2(samples->pattern[samples_index[h_rand]].membership);
		net->adjust_hessian(alg_estimation);	
	}
	net->copy_hessians();
}

//**********
//ucenie CNN 
void learn_cnn(WeakCNN *net, read_file* samples, long* samples_index, int number_of_samples, int iter, float &MSE, float &BPE) {
	
	float BPE_temp;
	float MSE_temp;	
	//BPE = 0.0
	//MSE = 0.0;	

	for (int h = 0 ; h < number_of_samples ; h++) {
						
		net->assign_input(samples, samples_index[h]);
		net->feed_forward();
		net->back_propagation_e(samples->pattern[samples_index[h]].membership, rbfPAT, MSE_temp, BPE_temp);
		net->adjust_weights(alg_ETA[iter], alg_MI[iter]);
		
		MSE += MSE_temp;
		BPE += BPE_temp;

	}
	MSE /= number_of_samples;
	BPE /= number_of_samples;
}

//********************************************
//vypocet chyby klasifikatora na celej mnozine 
void calculate_epsilon(WeakCNN* net, read_file* samples) {
	 
	double epsilon = 0.0;
	double pseudo = 0.0;
	int counter = 0;
	int *mapping = new int[255];
	
	//namapovanie indexov neuronov vystupnej vrstvy na list tried
	for (int c = 0 ; c < 255 ; c++) {
		mapping[c] = -1;
	}
	for (int i = 0 ; i < net->number_of_classes ; i++) {
		mapping[net->p_OUT->plane[i].membership] = i;
	}
	
	for (int h = 0 ; h < samples->patterns ; h++) {
		net->assign_input(samples, h);
		net->feed_forward();
		net->get_output(alg_radius);
		//urci sa vitazny neuron na vystupe a porovna s triedou
		if (alg_loss_function == 0) {
			if (samples->pattern[h].membership != net->temporal_class) {
				epsilon += samples->pattern[h].weightN;
				counter++;
			}
		}
		//pseudoloss verzia - na obyc neuronoch
		else if (alg_loss_function == 1) {
			//doplnit
		}
		//pseudoloss verzia - na RBF neuronoch
		else {	
			net->get_confidence(alg_radius);
			
			if (samples->pattern[h].membership != net->temporal_class) {
				counter++;
			}
		
			//zratava sa chyba epsilon cez mnozinu misslabels
			pseudo = 0.0;
			for (int i = 0 ; i < alg_classes ; i++) {
				if (mapping[i] >= 0){
					pseudo += samples->pattern[h].misslabelN_loc[i] * ((double)1.0 - net->p_OUT->plane[mapping[samples->pattern[h].membership]].conf + net->p_OUT->plane[mapping[i]].conf);
				}
			}
			epsilon += (double)0.5 * pseudo;
			
		}
	}
	net->misclassification = counter;
	net->epsilon = epsilon;
	
	delete[] mapping;
}

//********************************************
//pocitadlo tried 
void calculate_classes(read_file *samples) {
	int counter = 0;
	int *classes = new int[255];
	for (int i = 0 ; i < 255 ; i++) {
		classes[i] = 0;
	}
	//prechod cez vsetky klasifikatory v ensembli a budovanie pocetnosti
	pTemp = pHead;
	for (int t = 0 ; t < current_ensemble ; t++) {
		for (int i = 0 ; i < pTemp->classifier->number_of_classes ; i++) {
			classes[pTemp->classifier->classes[i]]++;
		}
		pTemp = pTemp->next;
	}
	// + kontrola tried v aktualnom datasete;
	for (int i = 0 ; i < 255 ; i++) {
		if (samples->classes[i] > 0) {
			classes[i]++;
		}
	}

	//vyskyt triedy predstavuje nenulova hodnota
	for (int i = 0 ; i < 255 ; i++) {
		if (classes[i] > 0){
			counter++;
			act_classes[i] = 1;
		}
		else {
			act_classes[i] = 0;
		}
	}
	delete[] classes;
	current_classes = counter;
}

//********************************************
//kopirovanie filtra zdroj->ciel 
void copy_filter(WeakCNN* s, WeakCNN* d) {
	
	char *info = new char[s->_CONVOLUTIONS + s->_SAMPLINGS + s->_CLASSICS +1];
	for (int i = 0 ; i < s->_CONVOLUTIONS + s->_SAMPLINGS + s->_CLASSICS +1 ; i++) {
		info[i] = s->info[i];
	}

	//filter predstavuje vsetky konvolucne a vzorkovacie vrstvy siete
	int convol = 0;
	int sampl = 0;
	int classic = 0;
	for (int a = 0 ; a < s->_CONVOLUTIONS + s->_SAMPLINGS ; a++) {
		switch (info[a]) {
			case 'c':
			for (int k = 0 ; k < s->p_C[convol]->this_planes ; k++) {
				//fprintf(f, "\n%f", p_C[convol]->plane[k].bias);	
				d->p_C[convol]->plane[k].bias = s->p_C[convol]->plane[k].bias;
				int max;
				if (convol == 0)
					max = 1;	//bereme zo vstupu
				else
					max = s->p_S[sampl-1]->this_planes;	//z predch. vrstvy
				for (int h = 0 ; h < max ; h++) {
					for (int i = 0 ; i < s->p_C[convol]->receptiveM ; i++) {
						for (int j = 0 ; j < s->p_C[convol]->receptiveN ; j++) {
							//fprintf(f, " %f", p_C[convol]->plane[k].w[h][i][j]);
							d->p_C[convol]->plane[k].w[h][i][j] = s->p_C[convol]->plane[k].w[h][i][j];
						}
					}
				}
			}
			convol++;
			break;
			case 's':
			for (int k = 0 ; k < s->p_S[sampl]->this_planes ; k++) {
				//fprintf(f, "\n%f", p_S[sampl]->plane[k].bias);
				d->p_S[sampl]->plane[k].bias = s->p_S[sampl]->plane[k].bias;
				//fprintf(f, " %f", p_S[sampl]->plane[k].w);
				d->p_S[sampl]->plane[k].w = s->p_S[sampl]->plane[k].w;

			}
			sampl++;
			break;
			/*
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
			*/
		}
	}


}

//********************************************
//prdanie klasifikatora do ensemblu
void add_classifier(int k, bool dataset_update) {
	pTemp = pAct;
	pAct = (ensemble *)malloc(sizeof(ensemble));
	pAct->classifier = new WeakCNN(classifier_structure[k][0],classifier_structure[k][1],classifier_structure[k][2],alg_act_function,alg_use_rbf_patterns);
	pAct->classifier->initialize_net(classifier_filename[k], true);
	current_ensemble++;

	if (pHead == NULL) {
		pHead = pAct;
		pHead->next = NULL;
		//nastavenie dataset flagu
		if (dataset_update == true) {
			pAct->classifier->dataset = 0;
		}
		else {
			pAct->classifier->dataset = 0;
		}

	}
	else {
		pTemp->next = pAct;	
		pAct->next = NULL;
		
		//nastavenie dataset flagu
		if (dataset_update == true) {
			pAct->classifier->dataset = pTemp->classifier->dataset + 1;
		}
		else {
			pAct->classifier->dataset = pTemp->classifier->dataset;
		}
	}
	

	//nastavenie rbf vah
	if (alg_use_rbf_patterns == 1) {
		pAct->classifier->initialize_rbf(rbfPAT);
	}
	
	//nastavenie pevnych masiek
	if (alg_builtin_extractor == 1) {
		pAct->classifier->apply_masks(builtinextractors);
	}

}

//********************************************
//prdanie silneho klasifikatora
void add_strong_classifier(int k) {
	sTemp = sAct;
	sAct = (s_ensemble *)malloc(sizeof(s_ensemble));
	sAct->classifier = new WeakCNN(classifier_structure[k][0],classifier_structure[k][1],classifier_structure[k][2],alg_act_function,alg_use_rbf_patterns);
	sAct->classifier->initialize_net(classifier_filename[k], true);
	strong_ensemble++;

	if (sHead == NULL) {
		sHead = sAct;
		sHead->next = NULL;
	}
	else {
		sTemp->next = sAct;	
		sAct->next = NULL;
	}

	//nastavenie rbf vah
	if (alg_use_rbf_patterns == 1) {
		sAct->classifier->initialize_rbf(rbfPAT);
	}
	
	//nastavenie pevnych masiek
	if (alg_builtin_extractor == 1) {
		sAct->classifier->apply_masks(builtinextractors);
	}
}

//********************************************
//vyradenie klasifikatora z ensemblu
void delete_classifier(ensemble *pDel) {
	delete pDel->classifier;
	pTemp = pHead;
	if (pDel != pHead) {
		while (pTemp->next != pDel) 
			pTemp = pTemp->next;
		pTemp->next = NULL;
	}
	if (pDel == pHead) {
		pHead = NULL;
	}
	free(pDel);
	pAct = pTemp;
	current_ensemble--;
}

//********************************************
//vyradenie klasifikatora z ensemblu
void delete_strong_classifier(s_ensemble *sDel) {
	delete sDel->classifier;
	sTemp = sHead;
	if (sDel != sHead) {
		while (sTemp->next != sDel) 
			sTemp = sTemp->next;
		sTemp->next = NULL;
	}
	if (sDel == sHead) {
		sHead = NULL;
	}
	free(sDel);
	sAct = sTemp;
	strong_ensemble--;
}


//********************************************
//inicializacia 
bool parameters() {
	
	char key;
	char temp_string[100];
	int temp_int;
	float temp_float;
	FILE *f, *g;
	
	strcpy(temp_string, working_directory);
	strcat(temp_string, "parameters");
	if( !(f=fopen(temp_string,"rt"))) {
		printf("\nParameters initialization failed !");
		return false;	
	}
	else {
		//globalne parametre algoritmu
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_iterations);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_datasets);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%f", &temp_float);
		
		if (temp_float > 1.0) {
			alg_samples = (int)temp_float;
			alg_part_samples = 0.0;
		}
		else {
			alg_samples = 0;
			alg_part_samples = temp_float;
		}

		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_estimation);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_learncycles);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%f", &alg_mse);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_filter_heredity);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_builtin_extractor);
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_classes);
		//aktivacna funkcia
		//1 - sigmoida
		//0 - tanh
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_act_function);
		//rbf cast siete
		//0 - bez rbf, na vystupe su single neurony
		//1 - s rbf patternami
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_use_rbf_patterns);
		fscanf(f, "%s", &temp_string);
		
		if (alg_use_rbf_patterns == 1) {
			rbfPAT = new rbfpatterns();
			strcpy(alg_rbf_patterns, working_directory);
			strcat(alg_rbf_patterns, temp_string);
			bool opened = rbfPAT->open_file(alg_rbf_patterns);
			if (opened == true) {
				rbfPAT->read_patterns(alg_act_function);
				printf("\nReading patterns for RBF layer completed !");
			}
			else {
				printf("\n\nError in reading file !");
				return false; 
			}	
		}
	
		// nastavenie ETA
		int t1;
		float t2;
		int max_cycles = 0;
		if (alg_filter_heredity > alg_learncycles) {
			max_cycles = alg_filter_heredity;
		}
		else {
			max_cycles = alg_learncycles;
		}
		alg_ETA = new float[max_cycles];
		fscanf(f, "%s", &temp_string);
		for (;;) {
			fscanf(f, "%d", &t1);
			if (t1 == -1) {
				break;
			}
			fscanf(f, "%f", &t2);
			for (int i = t1 ; i < max_cycles ; i++) {
				alg_ETA[i] = t2;
			}
		}

		//nastavenie MI
		alg_MI = new float[max_cycles];
		fscanf(f, "%s", &temp_string);
		for (;;) {
			fscanf(f, "%d", &t1);
			if (t1 == -1) {
				break;
			}
			fscanf(f, "%f", &t2);
			for (int i = t1 ; i < max_cycles ; i++) {
				alg_MI[i] = t2;
			}
		}
		
		//nazov suboru klasifikatora
		classifier_filename =  new char*[alg_datasets];
		fscanf(f, "%s", &temp_string);
		for (int i = 0 ; i < alg_datasets ; i++) {
			classifier_filename[i] = new char[255];
			fscanf(f, "%s", &temp_string);
			strcpy(classifier_filename[i],working_directory);
			strcat(classifier_filename[i],temp_string);
		}

		//struktura klasifikatora
		classifier_structure =  new int*[alg_datasets];
		fscanf(f, "%s", &temp_string);
		for (int i = 0 ; i < alg_datasets ; i++) {
			classifier_structure[i] = new int[3];
			fscanf(f, "%d", &classifier_structure[i][0]);
			fscanf(f, "%d", &classifier_structure[i][1]);
			fscanf(f, "%d", &classifier_structure[i][2]);
		}
		
		//nacitanie mien suborov datovych mnozin v zavyslosti od poctu
		dataset_names =  new char*[alg_datasets];
        fscanf(f, "%s", &temp_string);
		for (int i = 0 ; i < alg_datasets ; i++) {
			dataset_names[i] = new char[255];
			fscanf(f, "%s", &temp_string);
			strcpy(dataset_names[i], working_directory);
			strcat(dataset_names[i], temp_string);
		}
		
		//nacitanie klasifikatorv (ak su nejake naucene uz)
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &current_ensemble);
		//stare -> alokacia iba current ensemble
		for (int i = 0 ; i < current_ensemble  ; i++) {
			//zistenie struktury klasifikatora (naokolo - dalo by sa aj lepsie:)
			char file_number[5];
			char file_name[255];
			itoa(i,file_number,10);
			strcpy(file_name, working_directory);
			strcat(file_name, "cnn");
			strcat(file_name,file_number);
			strcat(file_name,".net");
			
			FILE *def;
			int CON, SAM, CLA;
			
			if( !(def=fopen(file_name,"rt"))) {
				printf("\nError in checking network structure of (%s) !", file_name);
				return false;
			}
			else {
				fscanf(def, "%s", &temp_string);
				fscanf(def, "\n%d %d %d", &CON, &SAM, &CLA);
				
				//postupna inicializacia
				pTemp = pAct;
				pAct = (ensemble *)malloc(sizeof(ensemble));
				pAct->classifier = new WeakCNN(CON, SAM, CLA, alg_act_function, alg_use_rbf_patterns);
				pAct->classifier->initialize_net(file_name, false);
				//pTemp->next = pAct;
				if (pHead == NULL) {
					pHead = pAct;
					pHead->next = NULL;
				}
				else {
					pTemp->next = pAct;	
					pAct->next = NULL;
				}			
			}		
		}
		
		//nacitanie klasifikatorv (ak su nejake naucene uz)
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &strong_ensemble);
		//stare -> alokacia iba konvolucnych filtrov
		for (int i = 0 ; i < strong_ensemble  ; i++) {
			//zistenie struktury klasifikatora (naokolo - dalo by sa aj lepsie:)
			char file_number[5];
			char file_name[255];
			itoa(i,file_number,10);
			strcpy(file_name, working_directory);
			strcat(file_name, "scnn");
			strcat(file_name,file_number);
			strcat(file_name,".net");
			
			FILE *def;
			int CON, SAM, CLA;
			
			if( !(def=fopen(file_name,"rt"))) {
				
				//strcpy(file_name, working_directory);
				//strcat(file_name, "scnn");
				//strcat(file_name,"0");
				//strcat(file_name,".net");
				//if( !(def=fopen(file_name,"rt"))) {
					printf("\nError in checking network structure of (%s) !", file_name);
					return false;
				//}
			}
			else {
				fscanf(def, "%s", &temp_string);
				fscanf(def, "\n%d %d %d", &CON, &SAM, &CLA);
				
				//postupna inicializacia
				sTemp = sAct;
				sAct = (s_ensemble *)malloc(sizeof(s_ensemble));
				sAct->classifier = new WeakCNN(CON, SAM, CLA, alg_act_function, alg_use_rbf_patterns);
				sAct->classifier->initialize_net(file_name, false);
				//sTemp->next = sAct;
				if (sHead == NULL) {
					sHead = sAct;
					sHead->next = NULL;
				}
				else {
					sTemp->next = sAct;	
					sAct->next = NULL;
				}			
			}		
		}


		//trenovanie alebo testovanie
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_procedure);
		
		//voting scheme
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_voting_scheme);
		
		//loss function
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%d", &alg_loss_function);
		
		//radius
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%f", &alg_radius);

		//rejection_treshold
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%f", &alg_rejection_treshold);
		
		//meno suboru dlzok
		fscanf(f, "%s", &temp_string);
		fscanf(f, "%s", &temp_string);
		
		alg_lenghts_file = new char[255];
		strcpy(alg_lenghts_file,working_directory);
		strcat(alg_lenghts_file,temp_string);
		
		//nacitanie masiek 
		if (alg_builtin_extractor == 1){
			
			builtinextractors = new masks();
			strcpy(temp_string, working_directory);
			strcat(temp_string, "masks");
			bool mask_open = builtinextractors->open_file(temp_string);
				
			if (mask_open == true) {
				builtinextractors->read_patterns();
				printf("\nReading masks completed !");
			}
		}

		fclose(f);
		printf("\nParameters initialization successfull !");
		
		return true;
	}
}

//*************************
//generator nahodnych cisel 

// constructor:
TRanrotBGenerator::TRanrotBGenerator(uint32 seed) {
  RandomInit(seed);
  // detect computer architecture
  union {double f; uint32 i[2];} convert;
  convert.f = 1.0;
  if (convert.i[1] == 0x3FF00000) Architecture = LITTLE_ENDIAN1;
  else if (convert.i[0] == 0x3FF00000) Architecture = BIG_ENDIAN1;
  else Architecture = NONIEEE;}


// returns a random number between 0 and 1:
double TRanrotBGenerator::Random() {
  uint32 x;
  // generate next random number
  x = randbuffer[p1] = _lrotl(randbuffer[p2], R1) + _lrotl(randbuffer[p1], R2);
  // rotate list pointers
  if (--p1 < 0) p1 = KK - 1;
  if (--p2 < 0) p2 = KK - 1;
  // perform self-test
  if (randbuffer[p1] == randbufcopy[0] &&
    memcmp(randbuffer, randbufcopy+KK-p1, KK*sizeof(uint32)) == 0) {
      // self-test failed
      if ((p2 + KK - p1) % KK != JJ) {
        // note: the way of printing error messages depends on system
        // In Windows you may use FatalAppExit
        printf("Random number generator not initialized");}
      else {
        printf("Random number generator returned to initial state");}
      exit(1);}
  // conversion to float:
  union {double f; uint32 i[2];} convert;
  switch (Architecture) {
  case LITTLE_ENDIAN1:
    convert.i[0] =  x << 20;
    convert.i[1] = (x >> 12) | 0x3FF00000;
    return convert.f - 1.0;
  case BIG_ENDIAN1:
    convert.i[1] =  x << 20;
    convert.i[0] = (x >> 12) | 0x3FF00000;
    return convert.f - 1.0;
  case NONIEEE: default:
  ;} 
  // This somewhat slower method works for all architectures, including 
  // non-IEEE floating point representation:
  return (double)x * (1./((double)(uint32)(-1L)+1.));}


// returns integer random number in desired interval:
int TRanrotBGenerator::IRandom(int min, int max) {
  int iinterval = max - min + 1;
  if (iinterval <= 0) return -0x80000000; // error
  int i = iinterval * Random(); // truncate
  if (i >= iinterval) i = iinterval-1;
  return min + i;}
  

void TRanrotBGenerator::RandomInit (uint32 seed) {
  // this function initializes the random number generator.
  int i;
  uint32 s = seed;

  // make random numbers and put them into the buffer
  for (i=0; i<KK; i++) {
    s = s * 2891336453 + 1;
    randbuffer[i] = s;}

  // check that the right data formats are used by compiler:
  union {
    double randp1;
    uint32 randbits[2];};
  randp1 = 1.5;
  assert(randbits[1]==0x3FF80000); // check that IEEE double precision float format used

  // initialize pointers to circular buffer
  p1 = 0;  p2 = JJ;
  // store state for self-test
  memcpy (randbufcopy, randbuffer, KK*sizeof(uint32));
  memcpy (randbufcopy+KK, randbuffer, KK*sizeof(uint32));
  // randomize some more
  for (i=0; i<9; i++) Random();
}
