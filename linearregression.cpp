#include "stdafx.h"
#include "linearregression.h"
#include <iostream>
LinearRegression::LinearRegression()
{}
LinearRegression::~LinearRegression()
{}
bool LinearRegression::learn()
{
	/*��ʼ�������õ����ֵ*/
	Regression::learn();
	/*1����Ȩ����ƫ��д��һ������ParamWB*/
	Eigen::MatrixXd beta;
	beta.resize(Weight.rows() + 1, 1);
	beta.block(0, 0, Weight.rows(), 1) = Weight;
	beta.block(Weight.rows(), 0, 1, 1) = Bias;
	/*������*/
	/*2����X���һ����� 1 */
	Eigen::MatrixXd X;
	X.resize(TrainningDataX.rows() + 1, TrainningDataX.cols());
	X.block(0, 0, TrainningDataX.rows(), TrainningDataX.cols()) = TrainningDataX;
	X.block(TrainningDataX.rows(), 0, 1, TrainningDataX.cols()) = Eigen::MatrixXd::Ones(1, TrainningDataX.cols());
	/*3����鷨:��������,�����L2���������С���˷����ParamWB*/
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