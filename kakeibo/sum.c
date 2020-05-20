#include <stdio.h>
#include "sum.h"

double sum(double payments[], int n)
{
    int i, total;
    total = 0;
 
    for(i = 0; i < n; i++) {
	total += payments[i];
    }
    return total;
}
