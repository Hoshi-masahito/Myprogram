#include<stdio.h>
#include"average.h"

double average(double payments[], int n) {
    int i;
    double total, ave;
    total = 0;

	for(i=0; i<n; i++) {
        total += payments[i];
    }
	ave = total / i;

	return ave;
}
