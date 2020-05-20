#include <stdio.h>
#include <string.h>
#include "sum.h"
#include "average.h"

#define N 256

int main(void)
{
    int i, c, len;
    char date[N], readline[N];
    double payments[N];
    FILE *fp, *fcp;

    printf("記録と削除のどちらを行いますか。\n記録ならば0、削除ならば1を入力してください。\n");
    scanf("%d", &c);

    if(c == 0) {
    fp = fopen("spending.txt", "a");
    if(fp == NULL) {
	printf("ファイルを開くことができませんでした。\n");
	return 0;
    }
	printf("日付を入力してください。\n");
	scanf("%s", date);

	fprintf(fp, "%s\n", date);

	i = 0;
	printf("各支出額を入力してください。¥n入力を終了する場合は0を入力してください。\n");
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

    } else if(c == 1) {

	printf("日付を入力してください。\n");
    	scanf("%s", date);

   	fp = fopen("spending.txt", "r");

        if(fp == NULL) {
	printf("ファイルを開くことができませんでした。\n");
	return 0;
	}

	fcp = fopen("tmp.txt", "w");

	while(fgets(readline, N, fp) != NULL) {
	    if(strstr(readline, date) == NULL) {
		fprintf(fcp, "%s", readline);
	    } else {
		fgets(readline, N, fp);
		fgets(readline, N, fp);
		fgets(readline, N, fp);
	    }
	}

	fclose(fp);
	fclose(fcp);

	fcp = fopen("spending.txt", "w");
	fcp = fopen("tmp.txt", "r");

	while(fgets(readline, N, fcp) != NULL) {
	    fprintf(fp, "%s", readline);
	}

	printf("日付: %sの記録を削除しました。\n", date);
	fclose(fp);
	fclose(fcp);
    } else {
	printf("作業を終了します。\n");
    }

    return 0;
}
