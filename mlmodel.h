#pragma once
#define MLMODEL_ASSERT(x) assert(x)
template<class DataType> class MLModel
{
	/******************construct******************/
public:
	MLModel()
	{}
	~MLModel()
	{}
	/******************communication******************/
	virtual void sendMsg(char * msg)
	{}
	/******************data******************/
public:
	virtual bool getTrainningData(DataType xData, DataType yData)
	{
		/*TrainningDataX:d*m����*,dΪ������,mΪ������/
		/*TrainningDataY:1*m����,mΪ������*/
		MLMODEL_ASSERT(isDataMatch(xData, yData));
		TrainningDataX = xData;
		TrainningDataY = yData;
		return true;
	}
	virtual bool getTestData(DataType xData, DataType yData)
	{
		/*TrainningDataX:d*m����*,dΪ������,mΪ������/
		/*TrainningDataY:1*m����,mΪ������*/
		MLMODEL_ASSERT(isDataMatch(xData, yData));
		TestDataX = xData;
		TestDataY = yData; 
		return true;
	}
	virtual bool isDataMatch(DataType &xData, DataType &yData)
	{
		return true;
	}
protected:
	DataType TrainningDataX;
	DataType TrainningDataY;
	DataType TestDataX;
	DataType TestDataY;
	/******************learning method******************/
public:
	virtual bool learn() = 0;
	/******************recognition method******************/
public:
	virtual DataType  recognize(DataType input) = 0;
};