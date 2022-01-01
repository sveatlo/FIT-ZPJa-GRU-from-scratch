#ifndef GRU_H
#define GRU_H

#include <map>
#include <random>
#include <string>
#include <vector>
#include "matrix.h"
#include "param.h"

using namespace std;

typedef struct {
	Matrix<double> r;
	Matrix<double> z;
	Matrix<double> h_hat;
	Matrix<double> h;
	Matrix<double> y;
	Matrix<double> x;
	Matrix<double> h_prev;
} GRU_step_data;

typedef struct {
	double loss;
	Matrix<double> h;
} GRU_forward_backward_return;

typedef struct {
	Matrix<double> dh_prev;
} GRU_backward_return;

typedef struct {
	vector<double> losses;
	map<string, Param> params;
} GRU_training_res;

class GRU {
public:
	GRU(map<char, unsigned> _char_to_idx, map<unsigned, char> _idx_to_char,  unsigned _vocab_size, unsigned _n_h = 100, unsigned _seq_len = 25);

	GRU_training_res train(vector<char> data, unsigned epochs, double lr);
	string sample(unsigned size, char seed = '\0');

private:
	map<char, unsigned> char_to_idx;
	map<unsigned, char> idx_to_char;
	unsigned vocab_size;
	unsigned n_h;
	unsigned seq_len;

	map<string, Param> params;

	double smooth_loss;
	default_random_engine sample_random_generator;

	Matrix<double> sigmoid(Matrix<double>);
	Matrix<double> softmax(Matrix<double>);
	Matrix<double> dsigmoid(Matrix<double>);
	Matrix<double> dtanh(Matrix<double>);
	void clip_grads();
	void reset_grads();
	void update_params(double lr);
	GRU_step_data forward_step(Matrix<double> x, Matrix<double> h_prev);
	GRU_backward_return backward_step(unsigned target_idx, Matrix<double> dh_next, Matrix<double> r, Matrix<double> z, Matrix<double> h_hat, Matrix<double> h, Matrix<double> y, Matrix<double> x, Matrix<double> h_prev);
	GRU_forward_backward_return forward_backward(vector<unsigned> x_batch, vector<unsigned> y_batch, Matrix<double> h_prev);
};

#endif
