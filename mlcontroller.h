#pragma once
#include "mlmodel.h"
template<class DataType> class MLController
{
/*construct*/
public:
	MLController()
	{}
	~MLController()
	{}

/*data*/
public:
	virtual void getTrainningData()
	{}
	virtual void getTestData()
	{}
protected:
	DataType TrainningData;
	DataType TestData;
	
/*learnning model*/
protected:
	MLModel<DataType> LearnningModel;

};