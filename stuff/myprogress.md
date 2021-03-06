# Прогресс в выполнение дипломной работы

## 1. Постановка проблемы (или как я понял о чем идет речь)

### Какую проблему решаем
Нужно для рекламодателя оценивать `рыночную цену`, в зависимости от `запроса`.

В `запрос` входит информация о пользователе, о показе рекламного объявления (на каком сайте и размер рекламного объявления) и содержимое рекламы ((?) что это именно не очень понял)

`Рыночная цена` вторая по величине ставка. О ней узнает только победитель аукциона

### Как решаем проблему

bidding logs ((?) - это же информация о пройденном аукционе)
Храниться как тройка значений *x* `запрос`, *b* `ставка`, *z* `рыночная цена`.

Основня задача - оценить плотность распределения рыночный цены, из полученного запроса.

#### Пункт 1
Обозначим вероятности.
Разбили величины ставков на интервалы и дальше считали вероятность события попадения `рыночной цены` в l-ый интервал
Используем теорию вероятности и определением функции выигрыша, и проигрыша и на их основе вероятность выигрыша.
#### Пункт 2
Нужна вероятность случайного события при условии запроса рекламодателя. Тут люди использовали `Recurrent Neural Networks` и получили вероясность события что рыночная цена принадлежит l-ому интервалу ставки, при условии запроса рекламодетеля.
#### Пункт 3
Решение задачи оптимизации. Ввели функции потерь и выбирают наилучший параметр для алгоритма в `пункте 2`.


### Что сделать еще
* [X] Разобраться в методе подсчета ошибки
* [X] Разобраться с Recurrent Neural Network приведенное в статье
* [X] Найти статьи решающие похожую задачу

## 2. Разобраться в реализации решения, приведенного в статье

### Что сделать тут
* [X] Для начала запустить, потыкать, подебажить
* [X] Посмотреть что за датасеты и тд
* [X] Разобраться где core, как написали сеть
* [X] Почитать статьи и написать на них обзор

### [Статья KN](http://apex.sjtu.edu.cn/public/files/members/20160929/functional-bid-lands.pdf) решение с помощью Decision Tree

Решали аналогичную задачу: найти плотность распределение рыночной цени и вероятность выиграша

то есть надо найти функцию $T_p(x)$ которая для заданого запроса $x$ найдет распределение цен $p_x(z) = T_p(x)$

Эту функцию ищу с помощью решающих деревьем.

Основная особенность в том что в разбиение используют неклассические функции (например энтропию или коэффициент Джини), а решают задачу кластаризации. (На данным момент про задачу кластеризации знаю почти ничего :disappoint:)

На каждом шаге (когда разбивают на поддеревью) максимизируют показатель  KL-Divergence $D_{KL}$, который в их формуле зависит от плотности распределения двух поддеревьев (формула есть в статье).

Используя задачу K-Means clustering строят деревья
TODO К чему пришли в конце и как получили распределения

### [DeepHit](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16160/15945) is a deep neural network model which predicts the probability of each bidding price from the minimum price to the maximum price

Статья никак не связана со ставками и рекламой. Но решают аналогичную задачу. Используют глубокое обучение, но с какими-то более хитрими нейтонками. 

TODO перечитать и дописать


### [DeepSurv](https://arxiv.org/pdf/1606.00931v3.pdf) is a Cox proportional hazard model with deep neural network  for feature extraction upon the sample covariates. The loss function is the negative partial likelihood of the winning and losing outcomes.

Дальше на почитать
- http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/win-price-pred.pdf

## 3. Подробный разбор статей

### Что сделать тут
* [X] Прочитать и разобрать статью решение задачи до использования ML
* [ ] Подробнее разобраться в моделе нейронки в основной статье
* [ ] Доделать обзор статьей в пункте выше

### Простой подход к решению задачи
Описание [Kaplan Meier](https://towardsdatascience.com/kaplan-meier-curves-c5768e349479), [статья решения](http://wnzhang.net/papers/unbias-kdd.pdf)

Сначала переводят лог данных из <b, z, w> в лог <b, d, n>. Чтобы свести задачу к формуле Kaplan–Meier. В которой survival function записывается как $$S(t) = \prod (1 - d_i \over n_i)$$ вероятность, что жизнь будет дольше чем t ((?) Как именно мы задачу survival analiz сводим на ставки).

Аналогично формуле KM, получаем вероятность проигрыша и выигрыша.

Дальше записиывают задачу минимизации (формула 6), после уравнение эмпирического риска (формула 7).

После как формулы потерь используют задачу логистической регресии. 
(?) Как я понял зачем. Логистическая регресия может дать апостериорную вероятность принадлежности объекта к классу. В этом случае задача классификации { -1, +1 } кликнут/не кликнут.

Потом написывают формулу для шага градиентного спуска.
И решешают во второй половине задачу оптимации кликов.

(?) Так а распределение вероятности то где???

### Подробнее про нейронку из основной статьи

Первый слой переводит набор аттрибутов в вектор [embedding layer](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12).

[Функция](https://github.com/SerTelnov/DLF/blob/master/python/BASE_MODEL.py#L244), где задается модель из статьи

[Статья](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537) разобра работы tensorflow с LSTM

(?) Что за time step? Это оно https://stats.stackexchange.com/questions/264546/difference-between-samples-time-steps-and-features-in-neural-network

## 4. Запуск кода на каком-то сервере

### Что сделать тут
* [X] Выбрать какой-то облачный сервер для запуска модели и разобраться в нем
* [X] Перевести код (хотя бы DLF) на python3
* [X] Запустить с настоящим датасетом
* [ ] Решить самому какую-то простую задачу с LSTM с использованием tensorflow и запушить на aws

Посмотрел на модный google colab. Для нашей задачи наверно не очень подходит, потому что у нас достаточно большой датасет (8.9 Гб) плюс он скорее всего будет не один.

Поднял сервер на AWS, потыкал что там есть, загрузил код

Надоело разбираться с приколами aws, текущая проблема была с python2, tensorflow и все другое работает с python3. В любом случае надо бы перевести на третий питон, а то стыд какой-то

Забил я его запускать на aws, мб потом разберусь. Как я понимаю что залить датасет на облако и из него брать данные, либо можно залить на drive и запустить в colab

Но как итог запустить просто на рабочем компьютере..

## 5. Разобраться в моделе DLF

* [ ] Прочитать статья про embedding, one-hot и тд
* [X] Основная цель - нарисовать граф нейронки 

Прочитал статьи про embedding
Статьи: 
* https://www.tensorflow.org/tutorials/text/word_embeddings
* https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
* https://towardsdatascience.com/how-to-create-word-embedding-in-tensorflow-ed0a61507dd0
* https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do

Надо посидеть и ручками по коду нарисовать модель, расписать каждую переменную, а то иначе страшно...

[tf lstm num_units](https://wszhan.github.io/2018/04/10/num_units_in_tf_lstm_rnn.html)
[еще lstm](https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)

[Как понял модель](https://user-images.githubusercontent.com/21954909/75664741-05257200-5c84-11ea-84ba-cf1ffd9b6460.JPG)


Что надо будет спросить
- Норм ли понял чет как.
- Спросить про про обучение в этой модели
- И про то что делать
- Норм ли протратить неделю чтобы переписать их код

## 6. Прочитать про Attention в RNN и Variational Autoencoders

Прочитать:

- http://kvfrans.com/variational-autoencoders-explained/
- https://www.youtube.com/watch?v=9zKuYvjFFS8

- https://pathmind.com/wiki/word2vec
- https://pathmind.com/wiki/attention-mechanism-memory-network
- https://towardsdatascience.com/an-introduction-to-attention-transformers-and-bert-part-1-da0e838c7cda
- https://blog.floydhub.com/attention-mechanism/amp/

## 7. Пишу свою модель

### Что сделать тут
* [X] Почитал документацию tensorflow, чтобы примерно понять как с ним работать.
* [X] Написать свою модель 
    (план капкан - копировать куски кода из модели стати и рефакторить с использованием нормальных подходов)
* [X] Подумать как можно быстро это тестировать

Параллельно буду писать свои идеи что плохо и можно улучшить

Первое что заметил это у tf немного странные embedding, не обычный какой-нибудь word2vec, а one-hot, который обучается 
dence слоем (?). Об этом [статья](https://data-flair.training/blogs/embedding-in-tensorflow/)
И нигде нет регуляризации

[Описание датасета](http://contest.ipinyou.com/ipinyou-dataset.pdf)
как с ним работали авторы статьи https://github.com/wnzhang/make-ipinyou-data, описание Benchmarking датасета https://arxiv.org/pdf/1407.7073.pdf

Разобрался как они работают с датасетом. [Собрали все фичи в один датасет](https://github.com/rk2900/make-ipinyou-data/blob/master/python/mkyx.py#L14) и переназвали все фичи через номер,
все номера в featindex.txt. А ето вообще норм?? (?) Больше они к featindex не обращаются, то есть только с ними и работают.

Подумал еще раз, вроде норм, но надо будет спросить.

(?) Сейчас проблема с подсчетом ошибки, выдаются слишком большое значение и че делать

### Идеи как можно улучшить
- Про embedding
    1. Мы обучаем модель для каждого рекламодателя отдельно, но ее используют как фичу номер 13. Точно ли это надо
    2. Слот ставок, их не так много, мб добавить еще? (и почему только до 100)
    3. Седьмая фича (slotid) выглядит как что-то странное. id может быть информативной фичой? Или это место размещение объевление? Тогда мб и норм :sweat_smile:
    4. Добавить batch-normalization на вход или в случае как сделал наш датасет это тупо?
- Подумать про функции активации. Я их брал с логикой __просто потому что__
- Сделать больше слоев (?)
- Добавить внимание (?)

[TF v.2 vs TF v.1](https://stackoverflow.com/questions/58441514/why-is-tensorflow-2-much-slower-than-tensorflow-1)


## 8. Добавляю механизм внимания

### Что сделать тут

* [ ] Прочитать про attention/self-attention
* [ ] Разобраться [статью](https://openreview.net/pdf?id=rJG8asRqKX) где использовали Attention в Survival analysis
* [ ] Реализовать attention в tensorflow
 
 
Хорошие статьи про self-attention 
 - https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
 - https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
 - https://medium.com/@Alibaba_Cloud/self-attention-mechanisms-in-natural-language-processing-9f28315ff905
 - https://blog.floydhub.com/attention-mechanism/
 - [публикация attention в переводе](https://arxiv.org/pdf/1409.0473.pdf)
 - https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
 
Статьи про описание LSTM в tensorflow
 -  https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
 - https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
 