#include <stdio.h>
#include "sum.h"
#include "average.h"

int main(void)
{
    int i;
    char date[256];
    double payments[256];
    FILE *fp;

    fp = fopen("spending.txt", "a");
    if(fp == NULL) {
	printf("ファイルを開くことができませんでした。\n");
	return 0;
    }
    printf("日付を入力してください。\n");
    scanf("%s", date);    
    fprintf(fp, "%s\n", date);


    i = 0;
    printf("各支出額を入力してください。\n入力を終了する場合は0を入力してください。\n");
    while(1) {
	scanf("%lf", &payments[i]);
	if(payments[i] == 0) {
	    break;
	};
	i++;
    }

    printf("sum: %lf\n", sum(payments, i));
    printf("average: %lf\n", average(payments, i));

    fprintf(fp, "総支出額: %lf円\n", sum(payments, i));
    fprintf(fp, "平均支出額: %lf円\n", average(payments, i));

    fprintf(fp, "\n");

    printf("総支出額と平均支出額をspending.txtに記録しました。\n");

    fclose(fp);
    return 0;
}
