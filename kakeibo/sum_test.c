#include <stdio.h>
#include "sum.h"

int main(void)
{
    int payments[] = {1, 2, 3};
    int n = sizeof(payments)/sizeof(payments[0]);

    if(sum(payments, n) == 6) {
	printf("%d\n", sum(payments, n));
	printf("テスト成功\n");
    } else {
	printf("%d\n", sum(payments, n));
	printf("テスト失敗\n");
    }

    return 0;
}
