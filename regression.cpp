#include "stdafx.h"
#include "regression.h"
#include <iostream>
Regression::Regression()
{}
Regression::~Regression()
{}
void Regression::sendMsg(char* msg)
{
	std::cout << msg << std::endl;
}
bool Regression::isDataMatch(Eigen::MatrixXd &xData, Eigen::MatrixXd &yData)
{
	if (xData.cols() == yData.cols())
		return true;
	return false;
}
bool Regression::learn()
{
	/*Ñ§Ï°²ÎÊý³õÊ¼»¯*/
	/*Wieght:d*1¾ØÕó*/
	/*Bias:1*1¾ØÕó*/
	Weight = Eigen::MatrixXd::Random(TrainningDataX.rows(), 1);
	Bias = Eigen::MatrixXd::Random(1, 1);
	return true;
}
Eigen::MatrixXd Regression::recognize(Eigen::MatrixXd input)
{
	/*input:d*m¾ØÕó*/
	/*output:1*m¾ØÕó*/
	/*Weight:d*1*/
	MLMODEL_ASSERT(input.rows() == Weight.rows());
	Eigen::MatrixXd output;
	output = (Weight.transpose())*input + Bias;
	return output;
}