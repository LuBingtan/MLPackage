#pragma once
#include "regression.h"

class LinearRegression : public Regression
{
	/******************construct******************/
public:
	LinearRegression();
	~LinearRegression();
	/******************data******************/
	/*ผฬณะ*/
	/******************optimize parameters******************/
	/*ผฬณะ*/	
	/******************learning method******************/
public:
	virtual bool learn() override;
	/******************recognition method******************/
	/*ผฬณะ*/
};