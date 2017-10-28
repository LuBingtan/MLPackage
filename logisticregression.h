#pragma once
#include "stdafx.h"
#include "regression.h"
class LogisticRegression : public Regression
{
	/******************construct******************/
public:
	LogisticRegression();
	~LogisticRegression();
	/******************data******************/
	/*�̳�*/
	/******************optimize parameters******************/
	/*�̳�*/
	/******************learning method******************/
public:
	virtual bool learn() override;
	Eigen::MatrixXd updateBeta(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd beta);
	Eigen::MatrixXd calP0(Eigen::MatrixXd Xi, Eigen::MatrixXd beta);
	Eigen::MatrixXd calDerivativeFst(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd beta);
	Eigen::MatrixXd calDerivativeSec(Eigen::MatrixXd X, Eigen::MatrixXd beta);
	/******************recognition method******************/
	/*�̳�*/
};