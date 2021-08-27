---
layout: post
title: "데이터 분석가가 Python에서 한번씩 보지만 궁금하지 않았던 것들"
author: "Chanjun Kim"
categories: Data분석
tags: [Python, tip, underbar, Asterisk, decorator, magickey, jupyter, f string]
image: 09_Python.png
---

## **학습목적**
데이터 분석가로써 가끔 한번씩 혹은 미쳐 몰랐던 파이썬의 기능들의 몇가지를 소개하고, 이해해보도록 한다.

- **목차**
    1. f string
    2. decorator
    3. underbar/under score
    4. Asterisk    
    5. jupyter magic key

---

### **F string**

- 분석을 하다보면 확인을 위해서 print한다던지, 혹은 query를 자동화한다던지 또는 어떤 데이터셋을 만들 때도 흔히 쓰이는 "".format을 많이 사용하게 됩니다. r에서는 paste나 sprintf를 사용하게 됩니다. 파이썬에서는 ~r보다 조금 자유롭고 편한하게(제 기준)~ 여러가지로 구현이 되어있습니다.
<br>
<br>
<br>
- 간단한 for문의 상태를 알 수 있는 print문과 sql문 등의 예시를 통해서 알아보도록 하겠습니다.


---

1. 가장 간단한 방식의 string + string 으로 표현합니다.
    - string의 + 와 int, float의 + 의 연산이 다르기 때문에 숫자를 str()로 다시 감싸주어야 사용할 수 있습니다.


```python
import os
import sys
```


```python
for i in range(101) :
    if i % 10 == 0 :
        print(str(i) + "번째 for문")
```

    0번째 for문
    10번째 for문
    20번째 for문
    30번째 for문
    40번째 for문
    50번째 for문
    60번째 for문
    70번째 for문
    80번째 for문
    90번째 for문
    100번째 for문


2. 조금 생소해보일 수도 있는 f string입니다. 
    - 간단한 작업을 할 때는 효율적이지만 복잡하거나 길어지면 가독성이나 코드가 간단하지는 않습니다.


```python
for i in range(101) :
    if i % 10 == 0 :
        print(f"{i}번째 for문")
```

    0번째 for문
    10번째 for문
    20번째 for문
    30번째 for문
    40번째 for문
    50번째 for문
    60번째 for문
    70번째 for문
    80번째 for문
    90번째 for문
    100번째 for문


3. 아마 가장 자주보는 방식의 string.format() 입니다.
    - 간단해 보이지만 응용할 수 있는 방법들이 다양하게 있습니다.


```python
for i in range(101) :
    if i % 10 == 0 :
        print("{}번째 for문".format(i))
```

    0번째 for문
    10번째 for문
    20번째 for문
    30번째 for문
    40번째 for문
    50번째 for문
    60번째 for문
    70번째 for문
    80번째 for문
    90번째 for문
    100번째 for문


- 어떠한 연속된 값 혹은 list의 값을 순서대로 formatting을 하고 싶으면 아래와 같은 방법으로 깔끔하게 해결할 수 있다.


```python
print("첫번째 값 : {} \n두번째 값 : {} \n세번째 값 : {}".format(0, 1, 2))
```

    첫번째 값 : 0 
    두번째 값 : 1 
    세번째 값 : 2



```python
print("첫번째 값 : {} \n두번째 값 : {} \n세번째 값 : {}".format(*[i for i in range(3)]))
```

    첫번째 값 : 0 
    두번째 값 : 1 
    세번째 값 : 2


- 그리고 어떤 값을 반복적으로 활용해야하는 상황에서 반복작업을 줄이기 위해서는 아래와 같이 key와 value를 매핑해주면 된다. dictionary를 활용할 때는 * 두개를 붙여 활용한다.


```python
print("첫번째 값 : {first_value} \n두번째 값 : {second_value} \n첫번째 값 AGAIN: {first_value} \n두번째 값 AGAIN : {second_value}".format(first_value = 1, second_value = 2))
```

    첫번째 값 : 1 
    두번째 값 : 2 
    첫번째 값 AGAIN: 1 
    두번째 값 AGAIN : 2



```python
print("첫번째 값 : {first_value} \n두번째 값 : {second_value} \n첫번째 값 AGAIN: {first_value} \n두번째 값 AGAIN : {second_value}".format(**{"first_value" : 1, "second_value" : 2}))
```

    첫번째 값 : 1 
    두번째 값 : 2 
    첫번째 값 AGAIN: 1 
    두번째 값 AGAIN : 2


- f"" 같은 경우는 간단하게 표현할 때 조금 더 직관적이고 간단해보인다.
- "".format() 같은 경우는 조금 복잡한 string을 만들 때(ex. sql 등)을 만들 때 활용하면 조금 더 효과적이다.

<br>
<br>
<br>

---

### **Decorator**

- 함수 위에 @표시로 되어있는 것을 데코레이터라고 합니다. 사실 데이터분석가가 decorator를 보는 일은 굉장히 드문 것 같습니다. 그런데 tensorflow를 사용하다보면 @tf.function 이라는 것을 심심치 않게 볼 수 있는데요, @가 의미하는 것이 무엇일지 알아보겠습니다.
<br>

---

- 기본적으로 decorator는 함수를 사용할 때마다 똑같은 기능을 함수에 넣기보다는 함수에 다시 씌워버리도록 하는 기능입니다.
    - 보통 코드가 얼마나 걸리는 지 확인하기 위해서 사용하는 코드를 데코레이터로 실행한 뒤 시간을 제보도록 하겠습니다.

- A부터 B까지의 합을 구하고 결과를 print해주는 함수를 만들어보겠습니다. 


```python
def sigma_fromA_toB(A, B) :
    print("result : {}".format(sum(range(A, B + 1))))
```


```python
sigma_fromA_toB(1,2)
```

    result : 3


- 보통 이 함수의 시간을 재기 위해서는 time 함수를 통하여 함수가 동작한 시간을 빼주게 됩니다.
<br>
1. 첫번째 방법은 보통은 테스트 용으로 활용하기 위해서 간단하게 함수 앞뒤에 time을 정의해주고 차이를 코드 실행시간으로 확인을 합니다.
    - 테스트하고 싶을 때만 앞뒤로 붙이면 되지만, 테스트할 경우가 많아지면 일일이 다 붙여넣기를 해주어야합니다.
2. 두번째 방법은 함수 안에 코드의 실행 시간을 print하게 기능을 넣었습니다.
    - 이 역시 필요한 함수에만 기능을 추가하면 되겠지만, 함수가 많아진다면 이 역시 매우 불편할 것입니다.


```python
import time
```


```python
stime = time.time()
sigma_fromA_toB(100000,1000000)
etime = time.time()
print("duringtime : {:.4f}".format(etime - stime))
```

    result : 495000550000
    duringtime : 0.0177



```python
def sigma_fromA_toB_toTime(A, B) :
    stime = time.time()
    print("result : {}".format(sum(range(A, B + 1))))
    etime = time.time()
    print("duringtime : {:.4f}".format(etime - stime))
```


```python
sigma_fromA_toB_toTime(100000,1000000)
```

    result : 495000550000
    duringtime : 0.0363


- 아래의 measure_time이라는 func를 중간에 실행하고 시간을 출력해주는 decorator 형식의 함수를 만들어 sigma_fromA_toB의 함수에 씌워보겠습니다.
<br>
<br>
- 결과는 아래와 같이 결과가 제대로 나오는 것을 볼 수 있습니다. 
    - 함수의 위 아래로 time을 정의할 필요가 없이 @measure_time이라고 붙여줬을 뿐인데 기능들이 실현될 수 있다는 것을 확인할 수 있습니다.
    - 앞으로 @ tf.function가 코드에 있더라도 tensorflow에서 잘 돌아갈 수 있도록 잘 정의해놓은 decorator function이라고 생각하고 쓰시면 될 것 같습니다.


```python
def measure_time(func) :
    def measure_time(*args, **kwargs) :
        stime = time.time()
        func(*args, **kwargs)
        etime = time.time()

        print("duringtime : {:.4f}".format(etime - stime))
    return measure_time
```


```python
@measure_time
def sigma_fromA_toB(A, B) :
    print("result : {}".format(sum(range(A, B + 1))))
```


```python
sigma_fromA_toB(100000,1000000)
```

    result : 495000550000
    duringtime : 0.0204


<br>
<br>
<br>

---

### **Under bar/ Under score**

- "_" 기호가 파이썬에 어디에서 나왔는지 궁금하신 분들도 있겠지만 생각보다 유용하고 자주 쓰인다는 것을 확인해보겠습니다.
<br>

---

### **언더바 하나만 사용되는 경우 1**
- 가장 자주 쓰이는 경우는 for문에서 딱히 쓸 필요가 없을 때 입니다.
    - 보통은 for loop 안의 index나 값을 쓰기 위해서 사용하지만 그냥 반복만 하면 될 경우입니다.


```python
for _ in range(10) :
    sigma_fromA_toB(100000,1000000)
```

    result : 495000550000
    duringtime : 0.0349
    result : 495000550000
    duringtime : 0.0222
    result : 495000550000
    duringtime : 0.0193
    result : 495000550000
    duringtime : 0.0630
    result : 495000550000
    duringtime : 0.0373
    result : 495000550000
    duringtime : 0.0484
    result : 495000550000
    duringtime : 0.0551
    result : 495000550000
    duringtime : 0.0290
    result : 495000550000
    duringtime : 0.0163
    result : 495000550000
    duringtime : 0.0154



```python
for _ in range(10) :
    sigma_fromA_toB(100000,1000000)
    print(f"_ size is {sys.getsizeof(_)}")
```

    result : 495000550000
    duringtime : 0.0177
    _ size is 24
    result : 495000550000
    duringtime : 0.0207
    _ size is 28
    result : 495000550000
    duringtime : 0.0269
    _ size is 28
    result : 495000550000
    duringtime : 0.0599
    _ size is 28
    result : 495000550000
    duringtime : 0.0328
    _ size is 28
    result : 495000550000
    duringtime : 0.0959
    _ size is 28
    result : 495000550000
    duringtime : 0.0295
    _ size is 28
    result : 495000550000
    duringtime : 0.0324
    _ size is 28
    result : 495000550000
    duringtime : 0.0161
    _ size is 28
    result : 495000550000
    duringtime : 0.0151
    _ size is 28


- 개인적인 의견이지만 "_"를 사용해도 memory에 저장이 안되는건 아니기 때문에 속도나 성능상 개선이 되진 않을 것 같습니다. 다만 의미상의 영역으로 보입니다. 그렇기 때문에 쓰지 않는 변수명을 적어도 성능이 저하될 것으로 보이진 않습니다.

### **언더바 하나만 사용되는 경우 2**
- Under bar는 최근 출력됬던 변수들(결과값)을 자동으로 저장하고 가지고 있습니다.
    1. a가 정의되고 한번 print되면 "_"에 최근 값이 저장이 됩니다.


```python
a = 1 + 1
a
```




    2




```python
_
```




    2




```python
b = 2 + 2
b
```




    4




```python
_
```




    4




```python
c = 3 + 3
```


```python
_
```




    4



<br>
<br>

### **언더바 위치와 개수에 따른 의미**
- 변수명에서 "_" 의 위치와 개수에 따라서 약속된 의미가 다릅니다. 사용자 정의 클래스나 함수를 import하는 과정에서 접근성에 대한 정의들입니다. 사실 제가 직접적으로 함수를 import할 일이 많지 않았기 때문에 와닿지 않아서 자세한 내용은 출처 블로그에서 확인하시길 바랍니다.
<br>

    - 앞에 하나의 언더바 \_variable : 내부 사용 코드, 즉 main문 안에서만 사용되는 함수
    - 뒤에 하나의 언더바 variable_ : 함수명이 겹칠 때 추가하여 중복을 방지(결국 그냥 어떤 특수문자를 붙이고 싶은데 _만 붙일 수 있다는 의미랑 비슷한 것 같습니다.)
    - 앞에 둘의 언더바 \_\_variable : 특정 변수가 override가 되는 것을 방지하기 위함, 즉 고유하게 유지하기 위한 방법(언제 쓰일지는 잘 모르겠습니다)
    - 앞과 뒤에 두개의 언더바 \_\_variable\_\_ : 매직메소드라고 불립니다. 가급적 이 방법을 직접 사용하는 것을 추천하지 않는다고 합니다. 즉 이미 만들어진 객체들이나 약속이 아니면 직접 이 형태의 객체를 만들지 않는 것이 좋다고 합니다. 보통 if __name__ == __main__, __init__을 할 때 많이 보입니다.


출처: https://eine.tistory.com/entry/파이썬에서-언더바언더스코어-의-의미와-역할 [아인스트라세의 SW 블로그]

<br>
<br>
<br>

---

### **Asterisk(\*표)**

- 기본적으로 곱셈이나 거듭제곱을 사용할 때 많이 보긴하지만 심심치 않게 func(\*args, \*\*kwargs) 이렇게 사용하는 함수들을 볼 수 있습니다. 이것이 의미하는 것과 왜 사용하는지에 대해서 알아보겠습니다.
<br>

<br>

#### 1. list 앞에 \*이 하나만 쓰일 때 : 기본적으로 리스트를 풀 때? 사용된다고 이해하시면 됩니다. 하지만 개별적으로 사용은 안됩니다.


```python
tmp1 = [1,2,3]
tmp2 = [4,5,6]
```


```python
[tmp1]
```




    [[1, 2, 3]]




```python
[*tmp1]
```




    [1, 2, 3]




```python
[tmp1, tmp2]
```




    [[1, 2, 3], [4, 5, 6]]




```python
[*tmp1, *tmp2]
```




    [1, 2, 3, 4, 5, 6]




```python
*tmp
```


      File "<ipython-input-53-ca40ad2e4bf3>", line 1
        *tmp
        ^
    SyntaxError: can't use starred expression here



<br>

#### 2. dict 앞에 \*이 하나만 쓰일 때 : dict를 list와 같이 풀 때는 \*를 두개 사용해주어야됩니다.


```python
tmp1 = {"a" : 1, "b" : 2, "c" : 3}
tmp2 = {"d" : 4, "e" : 5, "f" : 6}
```


```python
{**tmp1}
```




    {'a': 1, 'b': 2, 'c': 3}




```python
[tmp1]
```




    [{'a': 1, 'b': 2, 'c': 3}]




```python
[*tmp1]
```




    ['a', 'b', 'c']




```python
[**tmp1]
```


      File "<ipython-input-66-03dc013b6245>", line 1
        [**tmp1]
         ^
    SyntaxError: invalid syntax




```python
{**tmp1, **tmp2}
```




    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}




```python
[*tmp1, *tmp2]
```




    ['a', 'b', 'c', 'd', 'e', 'f']




```python
[**tmp1, **tmp2]
```


      File "<ipython-input-67-7655ba57a350>", line 1
        [**tmp1, **tmp2]
         ^
    SyntaxError: invalid syntax



<br>

#### 3. 위의 특징을 활용하여 함수에 적용하기


```python
def test_func(a = None, b = None, c = None) :
    print(f"a = {a}, b = {b}, c = {c}")
```


```python
test_func(1,2,3)
```

    a = 1, b = 2, c = 3


- 만일 [[1, 2, 3], [4,5,6], ["ㄱ", "ㄴ", "ㄷ"]] 이라는 list를 iterable하게 test_func에 적용해야하는 상황이 생겼을 때 어떻게 돌릴 수 있을까요?


```python
tmp = [[1, 2, 3], [4,5,6], ["ㄱ", "ㄴ", "ㄷ"]]
```


```python
for t in tmp :
    test_func(t[0], t[1], t[2])
```

    a = 1, b = 2, c = 3
    a = 4, b = 5, c = 6
    a = ㄱ, b = ㄴ, c = ㄷ


- 이런식으로 접근할 수도 있겠지만 위의 예시를 활용하여 Asterisk를 사용한다면 훨씬 간단하게 코드를 적용할 수 있습니다.


```python
for t in tmp :
    test_func(*t)
```

    a = 1, b = 2, c = 3
    a = 4, b = 5, c = 6
    a = ㄱ, b = ㄴ, c = ㄷ


- \*[]를 활용하면 간결한 코딩을 할 수 있지만 순서대로 들어가기 때문에 argument들이 많아지고 랜덤으로 부여됬을 때 순서를 다시 맞춰줘야하는 경우가 있습니다. 머신러닝/딥러닝을 하다보면 많은 하이퍼 파라미터가 있는데 batch_size, epoch, learning_rate같은 여러 하이퍼파라미터의 순서를 고려하는 것은 어려운 일이 될 것입니다. 이러한 경우는 dict를 활용하면 훨씬 간결해질 수 있습니다.


```python
def test_func(lr = 0.001, epoch = 100, batch_size = 64) :
    print(f"lr = {lr}, epoch = {epoch}, batch_size = {batch_size}")
```


```python
tmp1 = {"lr" : 0.000001, "epoch" : 200}
tmp2 = {"epoch" : 200, "batch_size" : 32}
tmp3 = {"epoch" : 200, "lr" : 0.000001, "batch_size" : 128}
```


```python
tmp = [tmp1, tmp2, tmp3]
```


```python
for t in tmp :
    test_func(**t)
```

    lr = 1e-06, epoch = 200, batch_size = 64
    lr = 0.001, epoch = 200, batch_size = 32
    lr = 1e-06, epoch = 200, batch_size = 128


- 앞에서 얘기했던 func(\*args, \*\*kwargs)은 결국에 list 형태의 파라미터는 \*args로 dict형식의 파라미터는 \*\*kwargs로 적용하는 것이다. kwargs는 keyword arguments로 key값이 있어 적용이 가능하다는 의미이다.

<br>
<br>
<br>

---

### **Jupyter Magic key**

- 파이썬을 활용하여 분석을 할 때 jupyter 환경을 사용하는 경우가 많은데, jupyter project는 분석에 필요한 패키지 관리 뿐 아니라 다양한 기능들을 magic key로 만들어 제공하고 있습니다. 대표적으로 코딩을 시작할 때 시작하는 %matplotlib inline도 jupyter magic key의 일부분입니다. 이렇듯 magic key는 %로 시작하게 됩니다.
- 시작 전에 어떤 기능들이 있는 지 알려주는 %lsmagic을 활용해서 확인해보겠습니다.
<br>

- magic키는 기본적으로 line magic key와 cell magic key로 나뉘게 됩니다. 이름을 듣고 유추할 수도 있겠지만, line magic key의 경우에는 % 하나로 이루어져있으며, 해당 매직키가 있는 라인의 코드에 대해서만 적용을 하게 되며, cell magic key는 cell 전체의 코드에 대해서 기능을 적용하게 됩니다.


```python
%lsmagic
```




    Available line magics:
    %alias  %alias_magic  %autoawait  %autocall  %automagic  %autosave  %bookmark  %cat  %cd  %clear  %colors  %conda  %config  %connect_info  %cp  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %lf  %lk  %ll  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %lx  %macro  %magic  %man  %matplotlib  %mkdir  %more  %mv  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %pip  %popd  %pprint  %precision  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %rep  %rerun  %reset  %reset_selective  %rm  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



- 위의 decorator 설명을 할 때도 코드의 실행 시간에 대해서 어떻게 코딩하는 가에 대한 부분이 있었는데 이런 부분도 매직키로 해결할 수 있습니다.


```python
%time time.sleep(5)
```

    CPU times: user 706 µs, sys: 1.65 ms, total: 2.36 ms
    Wall time: 5 s



```python
%time
time.sleep(3)
time.sleep(2)
```

    CPU times: user 3 µs, sys: 1 µs, total: 4 µs
    Wall time: 5.72 µs



```python
%%time 
time.sleep(2)
time.sleep(3)
```

    CPU times: user 948 µs, sys: 1.53 ms, total: 2.48 ms
    Wall time: 5 s


- 위에서 보듯이 똑같이 5초가 걸리게 실행했을 때 차이가 보입니다. line 매직키인 %time은 그 라인에 있는 코드에 대한 실행 시간을 측정하기 때문에 첫 코드는 5초의 실행시간을 뱉지만, 두번째 줄의 코드는 그 라인에 실행시킬것이 아무것도 없기 때문에 사실상 그냥 최소 시간을 측정하게 됩니다.
- 세번째의 cell 매직키인 %%time은 셀 전체의 시간을 측정하기 때문에 5초로 올바르게 측정하고 있습니다.
- 주의할 것은 매직키의 결과를 return해주는 것이 아니라 그냥 출력만 해주는 것입니다. %ls 등의 경로등을 변수로 사용할 때는 glob이나 sys 안에 있는 객체로 return해주는 함수를 활용해야합니다.
- 위에서 보듯 많은 기능들이 있지만 제가 보통 쓰는 것은 matplotlib, time 관련 매직키들이고 코랩을 활용할 경우 chdir, ls등을 사용할 경우가 많습니다. 보통 %matplotlib이 어디서 나온지 모른채 사용하는 경우가 많기 때문에 이번 기회에 %의 근본?에 대해서 한번 더 생각하는 기회가 됬으면 좋겠습니다.

---

code : https://github.com/Chanjun-kim/Chanjun-kim.github.io/blob/main/_ipynb/2021-08-08-PythonTip.ipynb

감사합니다.
