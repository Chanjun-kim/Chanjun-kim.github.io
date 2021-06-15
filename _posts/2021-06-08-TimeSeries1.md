---
layout: post
title: "시계열 데이터 - ARIMA"
author: "Chanjun Kim"
categories: Data분석
tags: [Data, TimeSeries, ARIMA, LSTM, BOOSTING, REGRESSION, 시계열데이터, 시계열분석]
image: 05_timeseries.png
---

## **학습목적**
시계열 데이터를 다루는 법과 시계열 예측을 하기 위한 여러가지 모델을 사용해보고 특성을 이해한다.<br>
이를 위해서 데이콘에서 진행 중인 대회인 전력사용량 예측 AI 경진대회의 데이터를 사용하고, 베이스라인 코드를 따라하고 또 나만의 코드로 만들어 결과를 제출하여 순위도 확인해볼 것이다.

### **시계열 데이터란?**
일정 시간 간격으로 배치된 데이터<br>
### **시계열 분석의 목표**
이런 시계열이 어떤 법칙에서 생성되는지 기본적인 질문을 이해하는 것<br> 
### **시계열 예측**
주어진 시계열을 보고 수학적인 모델을 만들어서 미래에 일어날 것들을 예측하는 것 _(X가 시계열 Y가 예측값)_<br>
일반적으로 이런 방법들은 금융시장에서의 주가 예측 등에서 많이 쓰인다.
> 출처 : https://ko.wikipedia.org/wiki/%EC%8B%9C%EA%B3%84%EC%97%B4


```python
import os
import sys
import warnings
from tqdm import tqdm

import itertools
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sns

import scipy
import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
```


```python
%matplotlib inline
warnings.filterwarnings("ignore")
```


```python
mpl.rcParams['axes.unicode_minus'] = False
# fm._rebuild()
plt.rcParams["font.family"] = 'NanumMyeongjo'
plt.rcParams["figure.figsize"] = (10,10)
```

---

**관련 단어 :**
- 계절성 : 특정 주기의 패턴을 가짐
- 주기성 : 특정 패턴을 가지나 일정한 주기를 갖지 않음
- 정상성 : 시계열 데이터가 추세나 계절성을 가지고 있지 않은 평균과 분산이 일정하여 시계열의 특징이 관측된 시간에 무관한 성질(주기성은 가질 수 있음) - <u>대부분의 시계열 데이터는 비정상성</u>
- 차분 : 평균이 일정하지 않은 시계열에 대하여 평균을 일정하게 만드는 작업
    - 일반 차분 : 전 시점의 자료의 차를 구하여 평균을 일정하게 만드는 방법
    - 계절 차분 : 계절성이 있는 자료에서 여러 시점의 차를 구하여 평균을 일정하게 만드는 방법
<br>
<br>
---

![Oops](https://otexts.com/fppkr/fpp_files/figure-html/stationary-1.png)

- 각 그림의 성질
    - 추세성 : a, c, e, f, i
    - 계절성 : d, h, i
    - 주기성 : g
    - 분산의 증가 : i
    - 정상성 : b, g


> 출처 : [https://otexts.com/fppkr/stationarity.html](https://otexts.com/fppkr/stationarity.html)

---

### **ARIMA(Autoregressvie integrated MovingAverage)**
**정의 :**
- 정상 시계열 데이터의 과거 관측값과 오차를 통하여 현재 시계열값을 예측하는 시계열 분석 기법 중 하나
<br>
<br>
- AR(Autoregression/자기상관) : 이전의 값이 이후의 값에 영향을 미치고 있는 상황
    - AR(1) 모델의 경우: yt = c + φ 1yt-1 + εt (−1 <ϕ1 <1)
    - AR(2) 모델의 경우: yt = c + φ 1yt-1 + φ 2yt-2 + εt
    <br>
    <br>
- I(Integrate/누적) : 차분을 이용
<br>
<br>
- MA(Moving Average/이동평균) : 랜덤 변수의 평균값이 지속적으로 증가하거나 감소하는 추세
    - ![image.png](attachment:fee6ef6a-56d8-4b8f-af58-f23226a4a065.png)![image.png](attachment:30328185-7194-451a-a850-b766c6c48640.png)
<br>
<br>
- ARIMA는 총 세개의 부분으로 이루어져 ARIMA(p, d, q)로 이루어진다.
    -  통상적으로 p + q < 2, p * q = 0 조건을 만족하는 모델을 사용한다고 한다.



```python
y = 1
coff = 0.3
const = 0
```


```python
ar_process = []
for i in range(300):
    error = np.random.randn()
    y = const + y * coff + error
    ar_process.append(y)
print(const / (1 - coff))
plt.plot(ar_process)
```

    0.0
    




    [<matplotlib.lines.Line2D at 0x28b820036d0>]




![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_10_2.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_10_2.png)


```python
ma_process = []
error = np.random.randn()
for i in range(300):
    ex_error = error
    error = np.random.randn()
    y = const + coff * ex_error + error
    ma_process.append(y)
plt.plot(ma_process)
```




    [<matplotlib.lines.Line2D at 0x28b82045be0>]




![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_12_1.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_11_1.png)


```python
train = pd.read_csv("data/dacon/energy/train.csv", encoding = "cp949")
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num</th>
      <th>date_time</th>
      <th>전력사용량(kWh)</th>
      <th>기온(°C)</th>
      <th>풍속(m/s)</th>
      <th>습도(%)</th>
      <th>강수량(mm)</th>
      <th>일조(hr)</th>
      <th>비전기냉방설비운영</th>
      <th>태양광보유</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2020-06-01 00</td>
      <td>8179.056</td>
      <td>17.6</td>
      <td>2.5</td>
      <td>92.0</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2020-06-01 01</td>
      <td>8135.640</td>
      <td>17.7</td>
      <td>2.9</td>
      <td>91.0</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2020-06-01 02</td>
      <td>8107.128</td>
      <td>17.5</td>
      <td>3.2</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2020-06-01 03</td>
      <td>8048.808</td>
      <td>17.1</td>
      <td>3.2</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2020-06-01 04</td>
      <td>8043.624</td>
      <td>17.0</td>
      <td>3.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 122400 entries, 0 to 122399
    Data columns (total 10 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   num         122400 non-null  int64  
     1   date_time   122400 non-null  object 
     2   전력사용량(kWh)  122400 non-null  float64
     3   기온(°C)      122400 non-null  float64
     4   풍속(m/s)     122400 non-null  float64
     5   습도(%)       122400 non-null  float64
     6   강수량(mm)     122400 non-null  float64
     7   일조(hr)      122400 non-null  float64
     8   비전기냉방설비운영   122400 non-null  float64
     9   태양광보유       122400 non-null  float64
    dtypes: float64(8), int64(1), object(1)
    memory usage: 9.3+ MB
    


```python
test = pd.read_csv("data/dacon/energy/test.csv", encoding = "cp949")
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num</th>
      <th>date_time</th>
      <th>기온(°C)</th>
      <th>풍속(m/s)</th>
      <th>습도(%)</th>
      <th>강수량(mm, 6시간)</th>
      <th>일조(hr, 3시간)</th>
      <th>비전기냉방설비운영</th>
      <th>태양광보유</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2020-08-25 00</td>
      <td>27.8</td>
      <td>1.5</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2020-08-25 01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2020-08-25 02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2020-08-25 03</td>
      <td>27.3</td>
      <td>1.1</td>
      <td>78.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2020-08-25 04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10080 entries, 0 to 10079
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   num           10080 non-null  int64  
     1   date_time     10080 non-null  object 
     2   기온(°C)        3360 non-null   float64
     3   풍속(m/s)       3360 non-null   float64
     4   습도(%)         3360 non-null   float64
     5   강수량(mm, 6시간)  1680 non-null   float64
     6   일조(hr, 3시간)   3360 non-null   float64
     7   비전기냉방설비운영     2296 non-null   float64
     8   태양광보유         1624 non-null   float64
    dtypes: float64(7), int64(1), object(1)
    memory usage: 708.9+ KB
    


```python
print(train.num.nunique())
print(test.num.nunique())
print(pd.concat([train.num.value_counts().sort_index(), test.num.value_counts()], axis = 1).head())
```

    60
    60
        num  num
    1  2040  168
    2  2040  168
    3  2040  168
    4  2040  168
    5  2040  168
    


```python
train.date_time = pd.to_datetime(train.date_time)
test.date_time = pd.to_datetime(test.date_time)
```


```python
train.date_time.describe()
```




    count                  122400
    unique                   2040
    top       2020-08-13 07:00:00
    freq                       60
    first     2020-06-01 00:00:00
    last      2020-08-24 23:00:00
    Name: date_time, dtype: object




```python
test.date_time.describe()
```




    count                   10080
    unique                    168
    top       2020-08-28 18:00:00
    freq                       60
    first     2020-08-25 00:00:00
    last      2020-08-31 23:00:00
    Name: date_time, dtype: object




```python
print(2040 * 60)
print(168 * 60)
```

    122400
    10080
    


```python
train.iloc[:,2:].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>전력사용량(kWh)</th>
      <th>기온(°C)</th>
      <th>풍속(m/s)</th>
      <th>습도(%)</th>
      <th>강수량(mm)</th>
      <th>일조(hr)</th>
      <th>비전기냉방설비운영</th>
      <th>태양광보유</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8179.056</td>
      <td>17.6</td>
      <td>2.5</td>
      <td>92.0</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8135.640</td>
      <td>17.7</td>
      <td>2.9</td>
      <td>91.0</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8107.128</td>
      <td>17.5</td>
      <td>3.2</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8048.808</td>
      <td>17.1</td>
      <td>3.2</td>
      <td>91.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8043.624</td>
      <td>17.0</td>
      <td>3.3</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_col = "전력사용량(kWh)"
```


```python
idxs = [1, 2, 3, 4]
(
    p9.ggplot() +
    p9.geom_line(data = train[train.num.isin(idxs)].assign(num = lambda x : x.num.astype(str)), mapping = p9.aes(x = "date_time", y = y_col, color = "num", group = "num")) +
    p9.theme_bw() +
    p9.theme(text = p9.element_text(family = "NanumMyeongjo"))
)
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_25_0.png)





    <ggplot: (174890195110)>



![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_23_0.png)

##### **AR, MA의 모수 추정**

- ACF(Autocorrelation function) : Lag에 따른 관측치들 사이의 관련성을 측정하는 함수
    - ![Oops](https://latex.codecogs.com/gif.latex?%5Crho_%7Bk%7D%3D%5Cfrac%7BCov%28y_%7Bt%7D%2C%20y_%7Bt+k%7D%29%7D%7BVar%28y_%7Bt%7D%29%7D)
<br>
<br>
- PACF(Partial autocorrelation function) : k 이외의 모든 다른 시점 관측치의 영향력을 배제하고 yt와 yt-k 두 관측치의 관련성을 측정하는 함수
    - ![Oops](https://latex.codecogs.com/gif.latex?%5Cphi_%7Bkk%7D%3Dcorr%28y_%7Bt%7D%2C%20y_%7Bt-k%7D%5Cmid%20y_%7Bt-1%7D%2C%20y_%7Bt-2%7D%2C%20...%2C%20y_%7Bt-k+1%7D%29)
<br>
<br>


- ![Oops](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTH1yl%2FbtqCSW7iSEV%2FUqrkvATr6vKA7vFwJIxN2K%2Fimg.png)
    - 사실 위에 정의를 정확히 이해하진 못하였으니 일단 이것만 기억하도록 한다.

> 출처 : [https://tjansry354.tistory.com/14](https://tjansry354.tistory.com/14) <br>
> 출처 : [https://kerpect.tistory.com/161](https://kerpect.tistory.com/161) <br>
> 출처 : [https://byeongkijeong.github.io/ARIMA-with-Python/](https://byeongkijeong.github.io/ARIMA-with-Python/)


```python
plt.rcParams["figure.figsize"] = (5, 3)
```

---

#### num이 1인 데이터의 plot과 acf, pacf 그래프를 그려보았다.
- 그림 상 보았을 때, acf 그래프가 사인 함수처럼 보이고 pacf가 1에서 급격히 떨어지는 것을 보아 아마 p = 1 이 적당하지 않을까 싶다.


```python
idx = 1
series = train[train.num == idx][y_col]
series.plot()
plot_acf(series)
plot_pacf(series)
plt.show()
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_30_0.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_30_1.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_30_2.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_27_0.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_27_1.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_27_2.png)

---

#### 차분을 1차 하여 num이 1인 데이터의 plot과 acf, pacf 그래프를 그려보았다.
- plot이 중심점에서 노이즈만 있는 것처럼 바뀌었고, acf, pacf 둘 다 0에서 급격히 떨어지것을 보아 p-q = 0이고 d = 1인 (1, 1, 1) 혹은 (2, 1, 2)가 적당하지 않을까 예상해본다.


```python
series_d1 = series - series.shift(1)
series_d1.plot()
plot_acf(series_d1[1:])
plot_pacf(series_d1[1:])
plt.show()
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_33_0.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_33_1.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_33_2.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_29_0.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_29_1.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_29_2.png)

---

#### 차분을 2차 하여 num이 1인 데이터의 plot과 acf, pacf 그래프를 그려보았다.
- plot은 차분을 안 한 데이터가 반전된 것처럼 보이고, acf, pacf는 거의 비슷하다.


```python
series_d2 = series - (2*series.shift(1)) - series.shift(2)
series_d2.plot()
plot_acf(series_d2[2:])
plot_pacf(series_d2[2:])
plt.show()
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_36_0.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_36_1.png)



![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_36_2.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_31_0.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_31_1.png) <br>
![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_31_2.png)


```python
cut = int(len(series) * 0.8)
train = series[:cut]
val = series[cut:]
```


```python
pdq_list = list(itertools.product(range(3), range(3), range(3)))
```


```python
pd_lists = list()
for pdq in tqdm(pdq_list[1:]) :
    arima = ARIMA(train, order = pdq)
    arima = arima.fit()
    predict = arima.predict(start = 1, end = len(val))
    aicc = arima.aicc
    aic = arima.aic
    bic = arima.bic
    mae = mean_absolute_error(val, predict)
    pd_lists = pd_lists + [[pdq, aicc, aic, bic, mae]]
```

    100%|███████████████████████████████████████████████████| 26/26 [00:08<00:00,  2.99it/s]
    


```python
arima_metric = pd.DataFrame(pd_lists, columns = ["pdq", "aicc", "aic", "bic", "mae"])
arima_metric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pdq</th>
      <th>aicc</th>
      <th>aic</th>
      <th>bic</th>
      <th>mae</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0, 0, 1)</td>
      <td>18897.962018</td>
      <td>18897.947276</td>
      <td>18914.139960</td>
      <td>149.379200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(0, 0, 2)</td>
      <td>18062.369967</td>
      <td>18062.345382</td>
      <td>18083.935628</td>
      <td>156.448449</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(0, 1, 0)</td>
      <td>16422.809294</td>
      <td>16422.806839</td>
      <td>16428.203787</td>
      <td>173.148794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0, 1, 1)</td>
      <td>16382.330861</td>
      <td>16382.323490</td>
      <td>16393.117387</td>
      <td>173.129768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0, 1, 2)</td>
      <td>16364.348722</td>
      <td>16364.333971</td>
      <td>16380.524817</td>
      <td>173.133983</td>
    </tr>
  </tbody>
</table>
</div>



- AICc, AIC, BIC는 모두 똑같은 상관관계를 갖게 되지만, 평가 metric인 mae와는 음의 상관관계를 보이고 있습니다.


```python
sns.heatmap(arima_metric.iloc[:, 1:].corr(), annot = True, fmt = ".2f", cmap = "Blues")
```




    <AxesSubplot:>




![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_43_1.png)


![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_37_1.png)

---

- aicc, aic, bic가 최소/최대가 되는 pdq의 조합은 동일하였습니다.
- 그러나 우리가 평가받아야될 mae는 최소/최대 조합이 달랐습니다.


```python
print(arima_metric[arima_metric["aicc"] == arima_metric["aicc"].max()])
print(arima_metric[arima_metric["aicc"] == arima_metric["aicc"].min()])
```

             pdq          aicc           aic          bic       mae
    0  (0, 0, 1)  18897.962018  18897.947276  18914.13996  149.3792
              pdq         aicc           aic           bic         mae
    19  (2, 0, 2)  16344.79814  16344.746448  16377.131817  170.946399
    


```python
print(arima_metric[arima_metric["aic"] == arima_metric["aic"].max()])
print(arima_metric[arima_metric["aic"] == arima_metric["aic"].min()])
```

             pdq          aicc           aic          bic       mae
    0  (0, 0, 1)  18897.962018  18897.947276  18914.13996  149.3792
              pdq         aicc           aic           bic         mae
    19  (2, 0, 2)  16344.79814  16344.746448  16377.131817  170.946399
    


```python
print(arima_metric[arima_metric["bic"] == arima_metric["bic"].max()])
print(arima_metric[arima_metric["bic"] == arima_metric["bic"].min()])
```

             pdq          aicc           aic          bic       mae
    0  (0, 0, 1)  18897.962018  18897.947276  18914.13996  149.3792
              pdq         aicc           aic           bic         mae
    19  (2, 0, 2)  16344.79814  16344.746448  16377.131817  170.946399
    


```python
print(arima_metric[arima_metric["mae"] == arima_metric["mae"].max()])
print(arima_metric[arima_metric["mae"] == arima_metric["mae"].min()])
```

             pdq          aicc           aic           bic         mae
    5  (0, 2, 0)  17802.388635  17802.386178  17807.782513  187.044582
             pdq          aicc           aic          bic       mae
    0  (0, 0, 1)  18897.962018  18897.947276  18914.13996  149.3792
    

---

- aicc, aic, bic가 최소값을 갖는 조합의 그래프입니다.


```python
arima = ARIMA(train, order = (2, 0, 2))
arima = arima.fit()
predict = arima.predict(start = 1, end = len(val))

predict.index = val.index

(
    p9.ggplot() +
    p9.geom_line(data = val.reset_index(), mapping = p9.aes(x = "index", y = y_col), group = 1, color = "green") +
    p9.geom_line(data = predict.reset_index(), mapping = p9.aes(x = "index", y = "predicted_mean"), group = 1, color = "red") +
    p9.theme(text = p9.element_text(family = "NanumMyeongjo"))
)
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_51_0.png)





    <ggplot: (174890630346)>



![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_44_0.png)

---

- aicc, aic, bic가 최대값 / mae가 최소값을 갖는 조합의 그래프입니다.


```python
arima = ARIMA(train, order = (0, 0, 1))
arima = arima.fit()
predict = arima.predict(start = 1, end = len(val))

predict.index = val.index

(
    p9.ggplot() +
    p9.geom_line(data = val.reset_index(), mapping = p9.aes(x = "index", y = y_col), group = 1, color = "green") +
    p9.geom_line(data = predict.reset_index(), mapping = p9.aes(x = "index", y = "predicted_mean"), group = 1, color = "red") +
    p9.theme(text = p9.element_text(family = "NanumMyeongjo"))
)
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_54_0.png)





    <ggplot: (174890669303)>



![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_46_0.png)

---

- mae가 최대값을 갖는 조합의 그래프입니다.


```python
arima = ARIMA(train, order = (0, 2, 0))
arima = arima.fit()
predict = arima.predict(start = 1, end = len(val))

predict.index = val.index

(
    p9.ggplot() +
    p9.geom_line(data = val.reset_index(), mapping = p9.aes(x = "index", y = y_col), group = 1, color = "green") +
    p9.geom_line(data = predict.reset_index(), mapping = p9.aes(x = "index", y = "predicted_mean"), group = 1, color = "red") +
    p9.theme(text = p9.element_text(family = "NanumMyeongjo"))
)
```


![png](2021-06-08-TimeSeries1_files/2021-06-08-TimeSeries1_57_0.png)





    <ggplot: (174890941386)>



![Oops](../assets/img/2021-06-08-TimeSeries1/2021-06-08-TimeSeries1_48_0.png)

---

- ARIMA의 모델 성능 지표인 AIC와 우리가 최소화시켜야할 MAE가 최소가 되는 점이 다르다는 것과 acf, pacf의 그래프에서 봤던 예상했던 결과와 전혀 다른 결과가 나왔습니다.
    - 대회가 아닌 실무에서는 무조건 mae가 낮다고 좋은 모델이 아닐 수 있다는 점을 유의하며 모델링을 해나가야겠습니다.

---

참고 자료 : 
- https://dacon.io/competitions/official/235736/codeshare/2628?page=1&dtype=recent
- https://byeongkijeong.github.io/ARIMA-with-Python/
- https://otexts.com/fppkr/arima-estimation.html
