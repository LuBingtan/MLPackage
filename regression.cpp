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
	/*ѧϰ������ʼ��*/
	/*Wieght:d*1����*/
	/*Bias:1*1����*/
	Weight = Eigen::MatrixXd::Random(TrainningDataX.rows(), 1);
	Bias = Eigen::MatrixXd::Random(1, 1);
	return true;
}
Eigen::MatrixXd Regression::recognize(Eigen::MatrixXd input)
{
	/*input:d*m����*/
	/*output:1*m����*/
	/*Weight:d*1*/
	MLMODEL_ASSERT(input.rows() == Weight.rows());
	Eigen::MatrixXd output;
	output = (Weight.transpose())*input + Bias;
	return output;
}