#include <stdio.h>
int init_int_arr(struct BigNum *target_arr, int *int_arr, int int_count, int *d_arr, int d_count, int sign);
int print_number(struct BigNum *target_arr);
int compare_abs_num(struct BigNum *num1, struct BigNum *num2);

struct BigNum {
	int i_digit[100];
	int d_digit[100];
	int i_total_digit;
	int d_total_digit;
	int sign;
};

int main() {

	int num1_int_arr[] = {1,4,7,8,9};
	int num1_d_arr[] = {0};
	int num1_sign = 0;
	struct BigNum num1;

	int num2_int_arr[] = { 4,7,8,9 };
	int num2_d_arr[] = { 0, 1 };
	int num2_sign = 0;
	struct BigNum num2;
	init_int_arr(&num1, num1_int_arr, sizeof(num1_int_arr)/sizeof(num1_int_arr[0]),
		num1_d_arr, sizeof(num1_d_arr)/sizeof(num1_d_arr[0]), num1_sign);
	init_int_arr(&num2, num2_int_arr, sizeof(num2_int_arr) / sizeof(num2_int_arr[0]),
		num2_d_arr, sizeof(num2_d_arr) / sizeof(num2_d_arr[0]), num2_sign);

	int compare_return = compare_abs_num(&num1, &num2);
	printf("Compare result : %d \n", compare_return);
	return 0;
}

int init_int_arr(struct BigNum *target_arr, int *int_arr, int int_count, int *d_arr, int d_count, int sign) {
	for (int i = 0; i < int_count; i++) {
		(target_arr->i_digit)[i] = int_arr[i];
	}
	for (int i = 0; i < d_count; i++) {
		(target_arr->d_digit)[i] = d_arr[i];
	}
	(target_arr->i_total_digit) = int_count;
	(target_arr->d_total_digit) = d_count;
	(target_arr->sign) = sign;
	return 0;
}

int print_number(struct BigNum *target_arr) {
	for (int i = 0; i < (target_arr->i_total_digit); i++) {
		printf("%d", (target_arr->i_digit)[i]);
	}
	printf(".");
	for (int i = 0; i < (target_arr->d_total_digit); i++) {
		printf("%d", (target_arr->d_digit)[i]);
	}
	printf("\n");
	return 0;
}

int compare_abs_num(struct BigNum *num1, struct BigNum *num2) {
	if ((num1->i_total_digit) > (num2->i_total_digit)) {
		return 1;
	}
	else if ((num1->i_total_digit) < (num2->i_total_digit)) {
		return 2;
	}
	else {
		for (int i = 0; i < (num1->i_total_digit); i++) {
			if ((num1->i_digit)[i] > (num2->i_digit)[i]) {
				return 1;
			}
			else if ((num1->i_digit)[i] < (num2->i_digit)[i]) {
				return 2;
			}
		}
		if ((num1->d_total_digit) > (num2->d_total_digit)) {
			for (int i = 0; i < (num2->d_total_digit); i++) {
				if ((num1->d_digit)[i] > (num2->d_digit)[i]) {
					return 1;
				}
				else if ((num1->d_digit)[i] < (num2->d_digit)[i]) {
					return 2;
				}
			}
			return 1;
		}
		else if ((num1->d_total_digit) < (num2->d_total_digit)) {
			for (int i = 0; i < (num1->d_total_digit); i++) {
				if ((num1->d_digit)[i] > (num2->d_digit)[i]) {
					return 1;
				}
				else if ((num1->d_digit)[i] < (num2->d_digit)[i]) {
					return 2;
				}
			}
			return 2;
		}
		else {
			for (int i = 0; i < (num1->d_total_digit); i++) {
				if ((num1->d_digit)[i] > (num2->d_digit)[i]) {
					return 1;
				}
				else if ((num1->d_digit)[i] < (num2->d_digit)[i]) {
					return 2;
				}
			}
			return 3;
		}
	}
	return 0;
}