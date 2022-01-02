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
        unsigned _vocab_size, unsigned _n_h, unsigned _seq_len, double _beta1, double _beta2)
    : char_to_idx(_char_to_idx), idx_to_char(_idx_to_char),
      vocab_size(_vocab_size), n_h(_n_h), seq_len(_seq_len), beta1(_beta1), beta2(_beta2) {

        // Xavier initialization
        double sdw = 1.0/ sqrt(this->vocab_size);
        double sdh = 1.0/ sqrt(this->n_h);

        // update gate - z
        this->params.insert(make_pair("Uz", Param("Uz", Matrix<double>::randn(this->n_h, this->n_h) * sdw + 0.5)));
        this->params.insert(make_pair("Wz", Param("Wz", Matrix<double>::randn(this->n_h, this->vocab_size) * sdh + 0.5)));
        this->params.insert(make_pair("bz", Param("bz", Matrix<double>(this->n_h, 1, 0))));

        // reset gate - r
        this->params.insert(make_pair("Ur", Param("Ur", Matrix<double>::randn(this->n_h, this->n_h) * sdw + 0.5)));
        this->params.insert(make_pair("Wr", Param("Wr", Matrix<double>::randn(this->n_h, this->vocab_size) * sdh + 0.5)));
        this->params.insert(make_pair("br", Param("br", Matrix<double>(this->n_h, 1, 0))));

        // hidden state / cell state / candidate activation vector / ... - h_hat
        this->params.insert(make_pair("Uh", Param("Uh", Matrix<double>::randn(this->n_h, this->n_h) * sdw + 0.5)));
        this->params.insert(make_pair("Wh", Param("Wh", Matrix<double>::randn(this->n_h, this->vocab_size) * sdh + 0.5)));
        this->params.insert(make_pair("bh", Param("bh", Matrix<double>(this->n_h, 1, 0))));

        // output (y) weight - V
        this->params.insert(make_pair("Wy", Param("Wy", Matrix<double>::randn(this->vocab_size, this->n_h) * sdh + 0.5)));
        this->params.insert(make_pair("by", Param("by", Matrix<double>::randn(this->vocab_size, 1) * sdh + 0.5)));

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
    // Matrix<double> e_x = x.exp();
    auto e_x = (x - x.max()).exp(); // substracting maximum for better numerical stability
    return e_x / (e_x.sum());
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

void GRU::update_params(double lr, int step) {
    for (auto &p : this->params) {
#define USE_ADAM 1
#if USE_ADAM
        auto tmp = p.second.d * (1 - this->beta1);
        p.second.m = (p.second.m * this->beta1) + tmp;

        tmp = p.second.d * p.second.d;
        tmp = tmp * (1 - this->beta2);
        p.second.w = (p.second.w * this->beta2) + tmp;

        auto m_corr = p.second.m / (1 - pow(this->beta1, step));
        auto w_corr = p.second.w / (1 - pow(this->beta2, step));

        tmp = w_corr.sqrt() + 1e-8;
        p.second.v -= ((m_corr * lr) / tmp);
#else
        p.second.m += p.second.d * p.second.d;
        auto tmp = (p.second.m + 1e-8).sqrt();
        p.second.v -= (p.second.d * lr) / tmp;
#endif
    }
}

GRU_step_data GRU::forward_step(Matrix<double> x, Matrix<double> h_prev) {
    assert(x.rows_n() == this->vocab_size && x.cols_n() == 1);
    assert(h_prev.rows_n() == this->n_h && h_prev.cols_n() == 1);

    Matrix<double> tmp;
    // z
    Matrix<double>& Uz = this->params["Uz"].v;
    Matrix<double>& Wz = this->params["Wz"].v;
    Matrix<double>& bz = this->params["bz"].v;
    // r
    Matrix<double>& Ur = this->params["Ur"].v;
    Matrix<double>& Wr = this->params["Wr"].v;
    Matrix<double>& br = this->params["br"].v;
    // h
    Matrix<double>& Uh = this->params["Uh"].v;
    Matrix<double>& Wh = this->params["Wh"].v;
    Matrix<double>& bh = this->params["bh"].v;
    // y
    Matrix<double>& Wy = this->params["Wy"].v;
    Matrix<double>& by = this->params["by"].v;

    // compute reset & update gate
    // z_t = sigmoid(Wz·x + Uz·h_prev + bz)
    tmp = Uz.mult(h_prev);
    auto z = this->sigmoid(Wz.mult(x) + tmp + bz);
    // r_t = sigmoid(Wr·x + Ur·h_prev + br)
    tmp = Ur.mult(h_prev);
    auto r = this->sigmoid(Wr.mult(x) + tmp + br);

    // compute hideen layers
    // h_hat_t = tanh(Wh·x + Uh·(r_t⊙h_prev)+bh)
    tmp = r*h_prev;
    tmp = Uh.mult(tmp) + bh;
    auto h_hat = (Wh.mult(x) + tmp).tanh();
    // h_t = z_t⊙h_prev + (1-z_t)⊙h_hat_t
    tmp = z*(-1) + 1; // (1-z_t)
    tmp = tmp*h_hat;
    auto h = (z*h_prev) + tmp;

    // compute regular output
    // y_t = Wy·h + by
    auto y = Wy.mult(h) + by;

    // compute probability distribution
    auto p = this->softmax(y);

    GRU_step_data step_data = {
        // main outputs
        .r = r,
        .z = z,
        .h_hat = h_hat,
        .h = h,
        .y = y,
        .p = p,
        // helpers
        .x = x,
        .h_prev = h_prev,
    };

    return step_data;
}

GRU_backward_return GRU::backward_step(
        unsigned target_idx,
        Matrix<double> dh_next,
        Matrix<double> r,
        Matrix<double> z,
        Matrix<double> h_hat,
        Matrix<double> h,
        Matrix<double> p,
        Matrix<double> x,
        Matrix<double> h_prev
){
    Matrix<double> tmp;
    Matrix<double>& Uz = this->params["Uz"].v;
    Matrix<double>& Ur = this->params["Ur"].v;
    Matrix<double>& Uh = this->params["Uh"].v;
    Matrix<double>& Wy = this->params["Wy"].v;

    auto hT = h.transpose();
    auto h_prevT = h_prev.transpose();
    auto xT = x.transpose();

    // ∂loss/∂y
    auto dy(p); dy(target_idx, 0) -= 1;

    // ∂loss/∂Wy and ∂loss/∂by
    this->params["Wy"].d += dy.mult(hT);
    this->params["by"].d += dy;

    auto dh = Wy.transpose().mult(dy) + dh_next;
    tmp = z*(-1) + 1; // (1-z_t)
    auto dh_hat = dh * tmp;
    tmp = this->dtanh(h_hat);
    auto dh_hat_l = dh_hat * tmp;

    // ∂loss/∂Wh, ∂loss/∂Uh and ∂loss/∂bh
    this->params["Wh"].d += dh_hat_l.mult(xT);
    tmp = (r * h_prev).transpose();
    this->params["Uh"].d += dh_hat_l.mult(tmp);
    this->params["bh"].d += dh_hat_l;

    auto drhp = Uh.transpose().mult(dh_hat_l);
    auto dr = drhp * h_prev;
    tmp = this->dsigmoid(r);
    auto dr_l = dr * tmp;

    // ∂loss/∂Wr, ∂loss/∂Ur and ∂loss/∂br
    this->params["Wr"].d += dr_l.mult(xT);
    this->params["Ur"].d += dr_l.mult(h_prevT);
    this->params["br"].d += dr_l;

    tmp = h_prev - h_hat;
    auto dz = dh * tmp;
    tmp = this->dsigmoid(z);
    auto dz_l = dz * tmp;

    // ∂loss/∂Wz, ∂loss/∂Uz and ∂loss/∂bz
    this->params["Wz"].d += dz_l.mult(xT);
    this->params["Uz"].d += dz_l.mult(h_prevT);
    this->params["bz"].d += dz_l;

    auto dh_fz_inner = Uz.transpose().mult(dz_l);
    auto dh_fz = dh * z;
    auto dh_fhh = drhp * r;
    auto dh_fr = Ur.transpose().mult(dr_l);

    auto dh_prev = dh_fz_inner + dh_fz + dh_fhh + dh_fr;

    return GRU_backward_return{
        .dh_prev = dh_prev,
    };
}

GRU_forward_backward_return GRU::forward_backward(vector<unsigned> x_batch, vector<unsigned> y_batch, Matrix<double> h_prev) {
    GRU_step_data init;
    init.h = h_prev;
    vector<GRU_step_data> progress = {init};

    double loss = 0;
    for (unsigned t = 0; t < this->seq_len; t++) {
        Matrix<double> x(this->vocab_size, 1, 0);
        x(x_batch[t], 0) = 1;

        GRU_step_data forward_res = this->forward_step(x, progress.back().h);

        progress.push_back(forward_res);

        // loss_t = true_label * log(predicted)
        // -> true_label=1
        //  => loss_t=1*log(...)
        // -> loss = -sum(loss_t)
        //  => loss_t = -1*log(...) & loss = sum(loss_t)
        loss += -1 * log(forward_res.p(y_batch[t], 0));
    }

    this->reset_grads();

    Matrix<double> dh_next(h_prev.rows_n(), h_prev.cols_n(), 0);
    for (unsigned t = this->seq_len; t > 0; t--) { // forward pass ended at index this->seq_len because it started at 1, not 0
        // cout << "backward step @ time t == " << t << endl;
        GRU_backward_return backward_res = this->backward_step(
            y_batch[t - 1], // chars in batch start from 0
            dh_next,
            progress.at(t).r,
            progress.at(t).z,
            progress.at(t).h_hat,
            progress.at(t).h,
            progress.at(t).p,
            progress.at(t).x,
            progress.at(t).h_prev
        );

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

    int step = 1;
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        cout << "Starting epoch no." << epoch << " with " << X.size() / this->seq_len
            << " batches" << endl;
        Matrix<double> h_prev(this->n_h, 1, 0);

        // int delete_n = 0;
        for (unsigned i = 0; i < X.size() - this->seq_len; i += this->seq_len, step++) {
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

            this->update_params(lr, step);
        }

        cout << endl;
        cout << "------------------------------ Epoch " << epoch << " ------------------------------" << endl;
        cout << "Loss: " << this->smooth_loss << endl;
        cout << "Sample: " << this->sample(250);
        cout << endl;
        cout << "---------------------------------------------------------------------" << endl;
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

        vector<double> probabilities = res.p.ravel();
        h = res.h;

        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        const unsigned idx = distribution(this->sample_random_generator);
        // std::vector<double>::iterator result;
        // result = std::max_element(probabilities.begin(), probabilities.end());
        // const unsigned idx = std::distance(probabilities.begin(), result);

        x.fill(0);
        x(idx, 0) = 1;

        sample += this->idx_to_char[idx];
    }

    return sample;
}
