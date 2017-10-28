// MLpackage.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mlcontroller.h"
#include "regression.h"
#include "linearregression.h"
#include "logisticregression.h"
#include <iostream>
#include <math.h>
#include <memory>
void testLinearRegression()
{
	Eigen::MatrixXd xData;
	xData.resize(1, 5);
	xData << 1.0000, 1.1000, 1.2000, 1.3000,1.4;
	Eigen::MatrixXd yData;
	yData.resize(1, 5);
	yData << 2.0000, 2.1000, 2.2000, 2.3000, 2.4000;
	std::cout << "xData:" << std::endl << xData << std::endl;
	std::cout << "yData:" << std::endl << yData << std::endl;
	std::shared_ptr<LinearRegression> test1(new LinearRegression());
	LinearRegression test;
	test.getTrainningData(xData, yData);
	std::cout << "diag:" << std::endl << Eigen::MatrixXd::Identity(5, 5) << std::endl;
	test.learn();
	Eigen::MatrixXd testData;
	testData.resize(1, 1); testData << 12;
	Eigen::MatrixXd xx = test.recognize(testData);
	std::cout << "testData:" << std::endl << testData << std::endl
		<< "RESULT:" << xx << std::endl;
}
void testLogisticRegression()
{
	Eigen::MatrixXd xData;
	xData.resize(1, 11);
	xData << -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5;
	Eigen::MatrixXd yData;
	yData.resize(1, 11);
	yData << 0.3775, 0.4013, 0.4256, 0.4502, 0.4750, 0.5000, 0.5250, 0.5498, 0.5744, 0.5987, 0.6225;
	LogisticRegression test;
	test.getTrainningData(xData, yData);
	test.learn();
}
int main()
{
	//testLinearRegression();
	testLogisticRegression();
	return 0;
}
