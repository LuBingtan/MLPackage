#include "stdafx.h"
#include "linearregression.h"
#include <iostream>
LinearRegression::LinearRegression()
{}
LinearRegression::~LinearRegression()
{}
bool LinearRegression::learn()
{
	/*初始化参数得到随机值*/
	Regression::learn();
	/*1、将权重与偏置写成一个变量ParamWB*/
	Eigen::MatrixXd beta;
	beta.resize(Weight.rows() + 1, 1);
	beta.block(0, 0, Weight.rows(), 1) = Weight;
	beta.block(Weight.rows(), 0, 1, 1) = Bias;
	/*求解参数*/
	/*2、在X最后一行添加 1 */
	Eigen::MatrixXd X;
	X.resize(TrainningDataX.rows() + 1, TrainningDataX.cols());
	X.block(0, 0, TrainningDataX.rows(), TrainningDataX.cols()) = TrainningDataX;
	X.block(TrainningDataX.rows(), 0, 1, TrainningDataX.cols()) = Eigen::MatrixXd::Ones(1, TrainningDataX.cols());
	/*3、领归法:若不满秩,则加入L2正则项，按最小二乘法求解ParamWB*/
	double lamda = 0;
	Eigen::MatrixXd XXT = X*(X.transpose());
	if (XXT.determinant() == 0)
	{
		lamda = 0.01;
	}
	XXT += lamda*Eigen::MatrixXd::Identity(X.rows(), X.rows());
	beta = (XXT.inverse())*X*(TrainningDataY.transpose());
	std::cout << "ParamWB:" << std::endl << beta << std::endl;
	Weight = beta.block(0, 0, beta.rows() - 1, 1);
	std::cout << "Weight:" << std::endl << Weight << std::endl;
	Bias = beta.block(beta.rows() - 1, 0, 1, 1);
	std::cout << "Bias:" << std::endl << Bias << std::endl;
	return true;
}