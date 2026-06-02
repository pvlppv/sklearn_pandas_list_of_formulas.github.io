# sklearn_pandas_list_of_formulas.github.io

Импорты — один раз в начало ноутбука:
```python
import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
```

---

## 1. Загрузка и осмотр
```python
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df.shape            # (строк, столбцов)
df.head()           # первые строки
df.describe()       # min/max/среднее по числовым
df.dtypes           # типы (object = категориальный текст)
df.nunique()        # число уникальных значений в каждом столбце
df['col'].value_counts()   # частоты категорий
```
**«В скольких столбцах среднее в сотнях (100–999)»:**
```python
m = df.mean(numeric_only=True)
print(((m >= 100) & (m < 1000)).sum())
```
**«Сколько столбцов имеют меньше 5 разных значений» (без target):**
```python
print((df.drop(columns=['target']).nunique() < 5).sum())
```

---

## 2. Маски: доля / количество  ← БАЗА
```python
(df['blue'] == 0).mean()    # ДОЛЯ (True=1,False=0) → когда просят "доля"
(df['blue'] == 0).sum()     # КОЛИЧЕСТВО True → когда просят "сколько"
```
Несколько условий: каждое в скобках, `&` = И (все сразу), `|` = ИЛИ (хотя бы одно).
```python
# доля при двух условиях
m = (df['2'] > df['2'].mean()) & (df['13'] < df['13'].median())
print(round(df.loc[m, 'target'].mean(), 2))

# "насколько больше A, чем B" среди фильтра
f = df[df['talk_time'] < 10]
print((f['dual_sim']==1).sum() - (f['dual_sim']==0).sum())

# счётчик с двумя условиями
print(((df['Pass']==0) & (df['education']=='high school')).sum())

# доля в подгруппе
sub = df[df['price_range']==2]
print(round((sub['blue']==0).mean(), 2))

# |разность долей| между группами
a = df[df['type']=='wild']['Pass'].mean()
b = df[df['type']=='domestic']['Pass'].mean()
print(round(abs(a-b), 2))
```

---

## 3. Новая колонка по условию
```python
# хотя бы одна опция → |
df['is_internet'] = ((df['three_g']==1)|(df['four_g']==1)|(df['wifi']==1)).astype(int)

# все три условия → &
df['Pass'] = ((df['score-1']>50)&(df['score-2']>50)&(df['score-3']>50)).astype(int)

# колонка-пара значений
df['cat_bio'] = '(' + df['type'] + ', ' + df['group'] + ')'

# новый числовой признак
df['ram_resolution'] = df['ram'] * (df['px_height'] + df['px_width'])
df['NEW'] = df['7'] * df['11']
```
Правило: `(условие).astype(int)` — скобки вокруг всего, `.astype(int)` снаружи (True/False → 1/0).

---

## 4. Пропуски (NaN)
```python
df.isna().sum()                  # пропусков по столбцам
(df.isna().sum() > 0).sum()      # сколько столбцов с пропусками

col = df.isna().sum().idxmax()   # столбец с НАИБОЛЬШИМ числом пропусков
mv = df[col].mean(); print(round(mv,1))
df[col] = df[col].fillna(mv)

col2 = df.isna().sum().replace(0, np.nan).idxmin()  # с НАИМЕНЬШИМ (но >0)
before = len(df); df = df.dropna(subset=[col2]); print(before-len(df))
```
Заполнить кат.→категорией, числ.→средним (и df, и test одинаково):
```python
n=0
for c in df.columns:
    if df[c].isna().any():
        if df[c].dtype==object: v='Unknown'
        else: v=df[c].mean()
        df[c]=df[c].fillna(v); test[c]=test[c].fillna(v); n+=1
print(n)
```

---

## 5. Корреляция / медиана / квартили
```python
c = df.corr(numeric_only=True)['price_range'].drop('price_range')
print(round(c[c.abs().idxmax()], 2))   # коэф. (со знаком) самого сильного по модулю

df['score-1'].median()
q1 = g.quantile(.25, interpolation='lower')   # lower — если просят "интерполяция меньшим"
q3 = g.quantile(.75, interpolation='lower')
print(q3 - q1)                                # IQR
```
Исправить опечатки в категориях + бинаризовать таргет:
```python
df['target'] = df['target'].replace({'Churn':1,'Not churn':0,'churn':1,'not churn':0})
```

---

## 6. Кодирование категорий
```python
# Label encoding всех текстовых
for c in df.select_dtypes('object').columns:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    test[c] = le.transform(test[c])
print(test.shape[1])     # сколько столбцов стало

# OneHot без мультиколлинеарности = drop_first=True
cat = df.select_dtypes('object').columns
dum = pd.get_dummies(df[cat], drop_first=True)
print(dum.shape[1])      # сколько числовых из категориальных

# Редкие категории (<1%) → самое частое, затем OHE drop='first'
print(df['touch_screen'].nunique())
vc = df['touch_screen'].value_counts(normalize=True)
rare = vc[vc < 0.01].index
mode = df['touch_screen'].mode()[0]
for d in (df, test): d['touch_screen'] = d['touch_screen'].replace(rare, mode)
```

---

## 7. Модель + кросс-валидация (СКЕЛЕТ)
```python
y = df['target']                  # имя таргета своё
X = df.drop(columns=['target'])

# логрегрессия, f1 на 3 фолдах
model = LogisticRegression(random_state=42)
print(round(cross_val_score(model, X, y, cv=3, scoring='f1').mean(), 2))

# дерево, roc_auc, глубина 5, энтропия
t = DecisionTreeClassifier(max_depth=5, criterion='entropy')
print(round(cross_val_score(t, X, y, cv=3, scoring='roc_auc').mean(), 1))
```
**scoring:** 2 класса f1 → `'f1'` · многоклассовый → `'f1_weighted'` · ROC → `'roc_auc'` · точность → `'accuracy'`

---

## 8. GridSearchCV (подбор C или глубины)
```python
grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}        # степени 10 от 0.001 до 100
gs = GridSearchCV(LogisticRegression(random_state=42), grid, cv=3, scoring='f1_weighted')
gs.fit(X, y)
print(gs.best_params_['C'])          # лучшее C
print(round(gs.best_score_, 2))      # лучшее качество

gd = {'max_depth': list(range(2, 21))}              # глубина 2..20
gs2 = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), gd, cv=3, scoring='roc_auc')
gs2.fit(X, y); print(gs2.best_params_['max_depth'])
```

---

## 9. Финал: «любая модель» → файл результата
```python
y = df['target']; X = df.drop(columns=['target'])
Xtest = test[X.columns]                  # те же признаки

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X, y)
pred = clf.predict(Xtest)                # метки классов
# pred = clf.predict_proba(Xtest)[:,1]   # вероятность (если нужна для ROC-AUC)

pd.DataFrame({'price_range': pred}).to_csv('result.csv', index=False)   # CSV с заголовком
pd.Series(pred).to_csv('result.txt', index=False, header=False)         # одна колонка без заголовка
```
Не взялся порог → попробуй `GradientBoostingClassifier()`. **Отправь файл ПОСЛЕДНИМ.**

---

## 10. Расчёты «в уме» → в Python
```python
# ROC-AUC по таблице (score, метка 0/1)
from sklearn.metrics import roc_auc_score
print(round(roc_auc_score([1,0,0,1,0], [0.7,0.6,0.3,0.45,0.92]), 2))

# Information Gain — РЕГРЕССИЯ (H = дисперсия ddof=1)
H = lambda a: np.var(a, ddof=1) if len(a)>1 else 0.0
yA=np.array([10,5,15,14]); yl=np.array([10,5,15]); yr=np.array([14])
print(round(H(yA) - len(yl)/len(yA)*H(yl) - len(yr)/len(yA)*H(yr), 1))

# Information Gain — КЛАССИФИКАЦИЯ (Джини)
g=lambda a,b:2*(a/(a+b))*(b/(a+b))
print(round(g(40,60) - 70/100*g(20,50) - 30/100*g(20,10), 2))

# Байес: P(гипотеза k | событие)
P=[0.4,0.35,0.25]; pe=[0.04,0.06,0.03]; k=1
print(round(P[k]*pe[k]/sum(p*e for p,e in zip(P,pe)), 2))

# Мат.ожидание: геометрич. E=1/p ; биномиальн. E=n*p
# word2vec: собрать вектор, в ответ (v**2).sum()
# SVM ширина полосы: 2/np.linalg.norm(w)
# ядро cos: np.exp(-np.sum((a-b)**2))  (при K(a,a)=1)
```
