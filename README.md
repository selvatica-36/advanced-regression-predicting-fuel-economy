# Predicting Fuel Efficiency 
[![Open in Visual Studio Code](https://img.shields.io/badge/Open%20in-Visual_Studio_Code-5C2D91?style=flat&logo=visual%20studio&logoColor=white)](https://open.vscode.dev/selvatica-36/predicting-fuel-economy) ![GitHub commit activity](https://img.shields.io/github/commit-activity/y/selvatica-36/predicting-fuel-economy)  ![GitHub last commit](https://img.shields.io/github/last-commit/selvatica-36/predicting-fuel-economy)  ![issues](https://img.shields.io/github/issues/selvatica-36/predicting-fuel-economy.svg) ![Python version](https://img.shields.io/badge/Python%20version-3.12.3-FF9900?style=flat&logo=python&logoColor=white)

## Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-298D46?style=for-the-badge&logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-5C2D91?style=for-the-badge) ![Git](https://img.shields.io/badge/Git-B1361E?style=for-the-badge&logo=git&logoColor=white) ![VSCode](https://img.shields.io/badge/VSCode-2962FF?style=for-the-badge&logo=visual%20studio&logoColor=white)

## 1. The Situation

*Ever wondered which car to buy?*
*Whether you should save for a more expensive car that is perhaps more **energy-efficient**, or just go for the cheaper one?*
*Will the cheaper one be more expensive in the long term because of its energy usage?*

We've all been there one way or the other, whether it is a car, a domestic appliance, or something else.

The trusted consumer magazine *InTheKnow* aims to provide you with well-researched and informed reports on issues you care about. And this is one of them.

For this reason, we have a hired a data science team to help us provide you the best solutions.

### Fuel economy

Fuel economy is a major factor in the cost of owning a car. We are studying the most influential characteristics (engine, car weight, etc.) that influence fuel economy for an upcoming article.

## 2. The Solution

I have built a regression model that help us understand and predict a car's fuel efficiency (mileage, mpg) based on its characteristics. 

This is based on a small dataset (398 records) from 1970-1982, a period of huge innovation in car's energy efficiency fue to the US Oil Crisis of 1971. As gas prices surged, car models with higher mileage (miles-per-gallon, mpg) were prefered by customers. The US Government also introduced regulations to drive the market towards more fuel-efficient engines during this period.

### 2.1. What are the most important car features to predict mpg?

- **Car weight** is a hugely important factor. Heavier cars are less fuel efficient.
- **Model year**: as time goes on, cars get more efficient on average. 
- **Origin**: american cars are the least fuel efficient on average in this dataset, whereas european and japanese cars tend to be more efficient. This makes historical sense:
  - During this time, Japanese cars took a larger market share because they were more fuel efficient.

### 2.3. Our final model 

The final model scored well on the test data, with an R-squared of 0.84 and a mean absolute error (MAE) of 2.11 miles-per-gallon.

## 3. Improvements and next steps
- Look into regularisation. Could try if Ridge regression improves the model.
- Sample size is small and outdated: try to find more recent and larger source(s) or car data, and see if this model generalises on current times.

## References
1. https://history.state.gov/milestones/1969-1976/oil-embargo
2. https://www.history.com/news/energy-crisis-1970s-innovation

## License
This is a public source repository.