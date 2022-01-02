#ifndef PARAM_H
#define PARAM_H

#include "matrix.h"

class Param {
	public:
		Param() {}
		Param(string name, Matrix<double> value) : v(value) {
			this->name = name;
			this->d = Matrix<double>(this->v.rows_n(), this->v.cols_n(), 0);
			this->m = Matrix<double>(this->v.rows_n(), this->v.cols_n(), 0);
			this->w = Matrix<double>(this->v.rows_n(), this->v.cols_n(), 0);
		}

		string name;
		Matrix<double> v; // value
		Matrix<double> d; // gradient
		Matrix<double> m; // mean (for adam)
		Matrix<double> w; // variance (for adam)
};


#endif
