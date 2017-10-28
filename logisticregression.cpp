#include "stdafx.h"
#include "logisticregression.h"
#include <iostream>
LogisticRegression::LogisticRegression()
{}
LogisticRegression::~LogisticRegression()
{}
bool LogisticRegression::learn()
{
	/*初始化参数得到随机值*/
	Regression::learn();
	/*1、将权重与偏置写成一个变量beta*/
	Eigen::MatrixXd beta; beta.resize(TrainningDataX.rows() + 1, 1);
	beta.block(0, 0, TrainningDataX.rows(), 1) = Weight;
	beta.block(TrainningDataX.rows(), 0, 1, 1) = Bias;
	std::cout << "beta:" << std::endl << beta << std::endl;
	/*求解参数*/
	/*2、在X最后一行添加 1 */
	Eigen::MatrixXd X; X.resize(TrainningDataX.rows() + 1, TrainningDataX.cols());
	X.block(0, 0, TrainningDataX.rows(), TrainningDataX.cols()) = TrainningDataX;
	X.block(TrainningDataX.rows(), 0, 1, TrainningDataX.cols()) = Eigen::MatrixXd::Ones(1, TrainningDataX.cols());
	std::cout << "X:" << std::endl << X << std::endl;
	std::cout << "Y:" << std::endl << TrainningDataY << std::endl;
	Eigen::MatrixXd betaPre = beta;
	Eigen::MatrixXd betaNext=beta;
	/*3、按牛顿法求解beta*/
	for (int time = 0; time < 100; time++)
	{
		betaPre = betaNext;
		betaNext = updateBeta(X, TrainningDataY, betaNext);
		if ((betaNext - betaPre).norm() < 1e-3)
		{
			break;
		}
	}
	Weight = betaNext.block(0, 0, betaNext.rows() - 1, 1);
	Bias = betaNext.block(betaNext.rows() - 1, 0, 1, 1);
	std::cout << "Weight:" << std::endl << Weight << std::endl;
	std::cout << "Bias:" << std::endl << Bias << std::endl;
	return true;
}
Eigen::MatrixXd LogisticRegression::updateBeta(Eigen::MatrixXd X,Eigen::MatrixXd Y,Eigen::MatrixXd beta)
{
	/*X:(d+1)*m*/
	/*Y:1*m*/
	/*beta:(d+1)*1*/
	Eigen::MatrixXd betaNext;
	betaNext.resize(beta.rows(), beta.cols());
	betaNext = beta - (calDerivativeSec(X, beta).inverse())*calDerivativeFst(X, Y, beta);
	return betaNext;
}
Eigen::MatrixXd LogisticRegression::calP0(Eigen::MatrixXd Xi, Eigen::MatrixXd beta)
{
	/*Xi:(d+1)*1*/
	/*beta:(d+1)*1*/
	Eigen::MatrixXd output(Eigen::MatrixXd::Zero(1,1));
	Eigen::MatrixXd temp1 = ((beta.transpose())*Xi).array();
	Eigen::MatrixXd temp = ((beta.transpose())*Xi).array().exp();
	output = 1 / (1 + ((beta.transpose())*Xi).array().exp());
	return output;
}
Eigen::MatrixXd LogisticRegression::calDerivativeFst(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd beta)
{
	/*X:(d+1)*m*/
	/*Y:1*m*/
	/*beta:(d+1)*1*/
	int m = X.cols();
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(X.rows(), 1);
	for (int i = 0; i < m; i++)
	{
		Eigen::MatrixXd P0 = calP0(X.block(0, i, X.rows(), 1), beta);
		Eigen::MatrixXd P1 = Eigen::MatrixXd::Ones(1, 1) - P0;
		Eigen::MatrixXd temp = X.block(0, i, X.rows(), 1)*(Y.block(0, i, 1, 1) - P1);
		output += X.block(0, i, X.rows(), 1)*(P1 - Y.block(0, i, 1, 1));
	}
	return output;
}
Eigen::MatrixXd LogisticRegression::calDerivativeSec(Eigen::MatrixXd X, Eigen::MatrixXd beta)
{
	/*X:(d+1)*m*/
	/*beta:(d+1)*1*/
	int m = X.cols();
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(X.rows(), X.rows());
	for (int i = 0; i < m; i++)
	{
		Eigen::MatrixXd P0 = calP0(X.block(0, i, X.rows(), 1), beta);
		Eigen::MatrixXd P1 = Eigen::MatrixXd::Ones(1, 1) - P0;
		output += X.block(0, i, X.rows(), 1)*(X.block(0, i, X.rows(), 1).transpose())*P1(0)*P0(0);
	}
	return output;
}