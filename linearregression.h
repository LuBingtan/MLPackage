#pragma once
#include "regression.h"

class LinearRegression : public Regression
{
	/******************construct******************/
public:
	LinearRegression();
	~LinearRegression();
	/******************data******************/
	/*�̳�*/
	/******************optimize parameters******************/
	/*�̳�*/	
	/******************learning method******************/
public:
	virtual bool learn() override;
	/******************recognition method******************/
	/*�̳�*/
};