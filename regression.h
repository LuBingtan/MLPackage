#pragma once
#include <Eigen\Eigen>
#include "mlmodel.h"
class Regression : public MLModel<Eigen::MatrixXd>
{
	/******************construct******************/
public:
	Regression();
	~Regression();
	/******************communication******************/
	virtual void sendMsg(char * msg) override;
	/******************data******************/
	virtual bool isDataMatch (Eigen::MatrixXd &xData, Eigen::MatrixXd &yData) override;
	/******************optimize parameters******************/
protected:
	Eigen::MatrixXd  Weight;
	Eigen::MatrixXd Bias;
	/******************learning method******************/
public:
	virtual bool learn() override;
	
	/******************recognition method******************/
public:
	virtual Eigen::MatrixXd recognize(Eigen::MatrixXd input) override;
};