# Solar-power-generation-forecast
태양광 발전량 예측 AI 경진대회

주최 : https://dacon.io/competitions/official/235680/overview/description/




## **model1 description**
##### author : hyeonho lee

1. lightgbm 
- preprocessing
> *1. table transform ( 7days columns )*  
> *2. remove nunique feature*

- modeling
> *1. feature importance ( >1 )*  
> *2. pinball loss ( using quantile objective function )*

- ensembles
> *1. 5folds*  
> *2. preprocessing applied differently*  
>- table transform ( 7, 6, 5, 4, 3 days)

- post process
> *1. 3시 30분 이전, 7시 30분 이후 발전량 0으로 고정*  
> *2. min valud : 0, max value : 100으로 수정*
