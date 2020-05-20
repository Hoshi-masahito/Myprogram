#include<stdio.h>
#include"average.h"

int main(void) {

	int payments[] =  {1, 2, 3};
	int n=sizeof(payments)/sizeof(payments[0]);

	if(average(payments, n) == 2) {
          printf("%d\n", average(payments, n));
	  printf("テスト成功\n");
	}else{
          printf("%d\n", average(payments, n));
	  printf("テスト失敗\n");
	}
	
	return 0;
}
