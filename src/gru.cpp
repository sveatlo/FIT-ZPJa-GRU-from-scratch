#include <algorithm>
#include <chrono>
#include <float.h>
#include <iostream>
#include <map>
#include <math.h>
#include <random>
#include <sstream>
#include <string>
#include "gru.h"
#include "matrix.h"

using namespace std;

GRU::GRU(map<char, unsigned> _char_to_idx, map<unsigned, char> _idx_to_char,
					 unsigned _vocab_size, unsigned _n_h, unsigned _seq_len)
		: char_to_idx(_char_to_idx), idx_to_char(_idx_to_char),
			vocab_size(_vocab_size), n_h(_n_h), seq_len(_seq_len) {

	// Xavier initialization
	double sd = 1.0/ sqrt(this->vocab_size + this->n_h);

	// update gate - z
	this->params.insert(make_pair("Uz", Param("Uz", Matrix<double>::randn(this->n_h, this->n_h) * sd + 0.5)));
	this->params.insert(make_pair("bz", Param("bz", Matrix<double>(this->n_h, 1, 0))));

	// reset gate - r
	this->params.insert(make_pair("Ur", Param("Ur", Matrix<double>::randn(this->n_h, this->n_h) * sd + 0.5)));
	this->params.insert(make_pair("br", Param("br", Matrix<double>(this->n_h, 1, 0))));

	// hidden state / cell state / candidate activation vector / ... - h_hat
	this->params.insert(make_pair("Uh", Param("Uh", Matrix<double>::randn(this->n_h, this->n_h) * sd + 0.5)));
	this->params.insert(make_pair("Wh", Param("Wh", Matrix<double>::randn(this->n_h, this->vocab_size) * sd + 0.5)));
	this->params.insert(make_pair("bh", Param("bh", Matrix<double>(this->n_h, 1, 0))));

    // output (y) weight - V
	this->params.insert(make_pair("V", Param("V", Matrix<double>::randn(this->vocab_size, this->n_h) * sd + 0.5)));

	this->smooth_loss = -1 * log(1.0f / this->vocab_size) * this->seq_len;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    this->sample_random_generator = default_random_engine(seed);
}

Matrix<double> GRU::sigmoid(Matrix<double> x) {
	// 1 / (1 + e^(-x))
	return ((x * -1).exp() + 1).divides(1);
}
Matrix<double> GRU::dsigmoid(Matrix<double> x) {
	// x * (1 - x)
	auto tmp = ((x * -1) + 1);
	return x * tmp;
}
Matrix<double> GRU::dtanh(Matrix<double> x) {
	// 1 - x*x
	return (x.pow(2) * -1) + 1;
}
Matrix<double> GRU::softmax(Matrix<double> x) {
	// e^x / sum(x)
	Matrix<double> e_x = x.exp();
	return e_x / (e_x.sum() + 1e-8);
}

void GRU::clip_grads() {
	for (auto &item : this->params) {
		this->params[item.first].d.clip(-5, 5);
	}
}

void GRU::reset_grads() {
	for (auto &item : this->params) {
		this->params[item.first].d = Matrix<double>(item.second.d.rows_n(), item.second.d.cols_n(), 0);
	}
}

void GRU::update_params(double lr) {
	for (auto &p : this->params) {
		p.second.m += p.second.d * p.second.d;

		Matrix<double> tmp = (p.second.m + 1e-8).sqrt();
		p.second.v += ((p.second.d * lr) / tmp) * -1;
	}
}

GRU_step_data GRU::forward_step(Matrix<double> x, Matrix<double> h_prev) {
    assert(x.rows_n() == this->vocab_size && x.cols_n() == 1);
    assert(h_prev.rows_n() == this->n_h && h_prev.cols_n() == 1);

	Matrix<double> tmp;
	Matrix<double>& Uz = this->params["Uz"].v;
	Matrix<double>& bz = this->params["bz"].v;
	Matrix<double>& Ur = this->params["Ur"].v;
	Matrix<double>& br = this->params["br"].v;
	Matrix<double>& Uh = this->params["Uh"].v;
	Matrix<double>& Wh = this->params["Wh"].v;
	Matrix<double>& bh = this->params["bh"].v;
	Matrix<double>& V = this->params["V"].v;

    // z_t = sigmoid(Uz·h_prev+bz)
    auto z = this->sigmoid(Uz.mult(h_prev) + bz);
    // r_t = sigmoid(Ur·h_prev+br)
    auto r = this->sigmoid(Ur.mult(h_prev) + br);
    // h_hat_t = tanh(Wh·x + Uh·(r_t⊙h_prev)+bh)
    tmp = r*h_prev;
    tmp = Uh.mult(tmp) + bh;
    auto h_hat = (Wh.mult(x) + tmp).tanh();
    // h_t = (1 - z_t)⊙h_prev+z_t⊙h_hat
    tmp = z*(-1) + 1;
    tmp = tmp*h_prev;
    auto tmp2 = (z*h_hat);
    auto h = tmp + tmp2;
    // y_t = softmax(V*h_t)
    auto y = this->softmax(V.mult(h));

	GRU_step_data step_data = {
        .r = r,
        .z = z,
        .h_hat = h_hat,
        .h = h,
        .y = y,
	};

	return step_data;
}

GRU_backward_return GRU::backward_step(){


    return GRU_backward_return{};
}

GRU_forward_backward_return GRU::forward_backward(vector<unsigned> x_batch, vector<unsigned> y_batch, Matrix<double> h_prev) {
	vector<GRU_step_data> progress = {GRU_step_data{ .h = h_prev }};

	double loss = 0;
	for (unsigned t = 0; t < this->seq_len; t++) {
		Matrix<double> x(this->vocab_size, 1, 0);
		x(x_batch[t], 0) = 1;

		GRU_step_data forward_res = this->forward_step(x, progress.back().h);

		progress.push_back(forward_res);

		loss += -1 * log(forward_res.y(y_batch[t], 0));
	}

	this->reset_grads();

	Matrix<double> dh_next(h_prev.rows_n(), h_prev.cols_n(), 0);

	for (unsigned t = this->seq_len; t > 0; t--) { // forward pass ended at index this->seq_len because it started at 1, not 0
		// cout << "backward step @ time t == " << t << endl;
		GRU_backward_return backward_res = this->backward_step();

		dh_next = backward_res.dh_prev;
	}
	this->clip_grads();

	return GRU_forward_backward_return{
			.loss = loss,
			.h = progress.back().h,
	};
}

GRU_training_res GRU::train(vector<char> _X, unsigned epochs, double lr = 0.001) {
	int num_batches = _X.size() / this->seq_len;
	vector<char> X(_X.begin(), _X.begin() + num_batches * this->seq_len);
	vector<double> losses;

	for (unsigned epoch = 0; epoch < epochs; epoch++) {
		cout << "Starting epoch no." << epoch << " with " << X.size() / this->seq_len
				 << " batches" << endl;
		Matrix<double> h_prev(this->n_h, 1, 0);

		// int delete_n = 0;
		for (unsigned i = 0; i < X.size() - this->seq_len; i += this->seq_len) {
			int batch_num = epoch * epochs + i / this->seq_len;
			cout << "\rEpoch " << epoch << ": batch " << batch_num << "/" << X.size() / this->seq_len << " (loss: " << this->smooth_loss << ")";
			cout.flush();


			// prepare data
			vector<unsigned> x_batch, y_batch;
			for (unsigned j = i; j < i + this->seq_len; j++) {
				char c = X[j];
				x_batch.push_back(this->char_to_idx[c]);
			}
			for (unsigned j = i + 1; j < i + this->seq_len + 1; j++) {
				char c = X[j];
				y_batch.push_back(this->char_to_idx[c]);
			}

			// forward-backward on batch
			GRU_forward_backward_return batch_res = this->forward_backward(x_batch, y_batch, h_prev);

			// this->smooth_loss = batch_res.loss;
			this->smooth_loss = this->smooth_loss * 0.99 + batch_res.loss * 0.01;
			losses.push_back(this->smooth_loss);

			this->update_params(lr);
		}

		cout << endl;
		cout << "---------------Epoch " << epoch << "----------------------------"
				 << endl;
		cout << "Loss: " << this->smooth_loss << endl;
		cout << "Sample: " << this->sample(100, 't');
		cout << endl;
		cout << "--------------------------------------------------" << endl;
	}

	// return make_pair(losses, this->params);
	return GRU_training_res{
			.losses = losses,
			.params = this->params,
	};
}

string GRU::sample(unsigned size, char seed) {
	Matrix<double> x(this->vocab_size, 1, 0);
	if (seed != '\0') {
		x(this->char_to_idx[seed], 0) = 1;
	}

	Matrix<double> h(this->n_h, 1, 0);

	string sample = "";
	for (unsigned i = 0; i < size; i++) {
		GRU_step_data res = this->forward_step(x, h);

		vector<double> probabilities = res.y.ravel();
		h = res.h;

		std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
		const unsigned idx = distribution(this->sample_random_generator);

		x.fill(0);
		x(idx, 0) = 1;

		sample += this->idx_to_char[idx];
	}

	return sample;
}
