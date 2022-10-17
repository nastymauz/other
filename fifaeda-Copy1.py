#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import matplotlib.pyplot as plt # visualization
import numpy as np # linear algebra
import pandas as pd # data processing
import plotly.figure_factory as ff
import plotly.express as px


# # Домашнаяя работа: работа с numpy, pandas и matplotlib

# В этой домашней работе вам предстоит выполнить EDA (Exploratory Data Analysis) датасета с данными о футболистах FIFA.
# 
# Требования к выполнению домашней работы:
# - Во всех графиках должны быть подписи через `title`, `legend`, etc.
# - Убедитесь, что после сохранения ноутбука графики всё ещё видно, если открыть ноутбук заново. Если не видно - напишите в общий чатик, вам помогут либо преподаватели, либо те, кто уже столкнулся с этой проблемой
# - Можно баловаться с цветами, но в меру. Если в итоге работа станет нечитаемой, то задание не будет засчитано
# - Если вы сдаете работу в Google Colaboratory, убедитесь, что ваша тетрадка доступна по ссылке. Если в итоге по каким-то причинам тетрадка не будет открываться у преподавателя, задание не будет засчитано

# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


df[['Wage']].head(10)


# # Работа с датафреймом

# Заполните пропуски:
# 
# 1. В датасете `88` переменных 
# 2. Из них числовых `43`

# # Работа с графикой

# 3. Изучите распределение возрастов с помощью колонки Age [px.histogram](https://plotly.com/python/histograms/)
# 4. Изучите [более продвинутые методы](https://plotly.com/python/distplot/) визуализации распределений 

# In[6]:


df['Age'].hist(range=(0,45), bins=30)
plt.title('Распределение возрастов')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()


# In[7]:




np.random.seed(1)

x = df['Age']
hist_data = [x]
group_labels = ['Возраст']

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Распределение возрастов')
fig.show()


# 5. Найдите колонку, показывающую, является ли футболист левой или правшой
# 6. Визуализируйте соотношенеие между левшами и правшами (считаем, что признак "левша"/"правша" определяет предпочтительную ногу для ударов

# In[8]:


fig = px.histogram(df, x="Preferred Foot")
fig.update_layout(title_text='Соотношенеие между левшами и правшами')
fig.show()


# 7. Визуализируйте Международный Рейтинг футболистов (International Reputation)
# 8. Ассоциирован ли он с национальностью? С футбольным клубом? 
# 
# ![Игроки](Players.png "Title")

# In[9]:


fig = px.histogram(df, x="International Reputation")
fig.update_layout(title_text='Международный Рейтинг футболистов')
fig.show()


# In[10]:


(df
 .pivot_table(index='Nationality', columns='International Reputation', values='Club', aggfunc='count')
 .sort_values([1,2,3,4,5], ascending = False))


# In[11]:


(df
 .pivot_table(index='Club', columns='International Reputation', values='ID', aggfunc='count')
 .sort_values([5], ascending = False))


# In[12]:


df.groupby('International Reputation')['Club'].count().reset_index()


# Футболистов с высоким Международным Рейтингом очень мало, относительно общего числа футболистов.

# In[13]:


df45 = df[df['International Reputation'] > 3]


# In[14]:


(df45
 .pivot_table(index='Nationality', columns='International Reputation', values='Club', aggfunc='count')
 .sort_values([4,5], ascending = False))


# In[15]:


(df45
 .pivot_table(index=['Club','Nationality'], columns='International Reputation', values='ID', aggfunc='count')
 )


# In[16]:


len(df['Club'].unique())


# Международный Рейтинг футболистов не ассоциирован с национальностью. 
# 
# Есть топовые клубы, в которых несколько высокорейтинговых футболистов, но футболистов с рейтингом 5 всего 6 среди почти 2 тысяч и в них нет явного преобладания какой-либо национальности. 

# 9. Есть ли зависимость между потенциалом (Potential) и Международным Рейтингом футболистов (International Reputation)? Визуализируте её
# 10. Влияет ли на потенциал игровая нога? Почему так может случиться? (картинка в тему)
# 
# ![Коазуация или корреляция?](causuation.png "???")

# In[17]:


df.plot(x='International Reputation', y='Potential', kind='scatter', alpha=0.1, figsize=(15,8), title='Зависимость между потенциалом и Международным Рейтингом футболистов')
print('Коэффициент корреляции равен:', df['Strength'].corr(df['Stamina']))


# Коэффициент корреляции этих двух показателей 0.26, это говорит о слабой зависимости.

# In[18]:


(df
 .pivot_table(index='Potential', columns='Preferred Foot', values='ID', aggfunc='count')
 .sort_values(['Potential'], ascending = False))


# По данным можно сказать, что самый высокий потенциал чаще всего у правшей, но правшей в мире больше(в мире всего 15% левшей), поэтому футболисты с правой игровой ногой чаще имеют высокий потенциал и игровая нога не влияет

# Теперь пройдёмся по показателям игроков...
# 
# 11. Есть ли зависимость между **выносливостью** игрока (Stamina) и его **силой** (Strength)? Какая? Точно ли это зависимость? 
# 

# In[19]:


df.plot(x='Stamina', y='Strength', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость между выносливостью игрока и его силой')
print('Коэффициент корреляции равен:', df['Strength'].corr(df['Stamina']))


# Коэффициент корреляции этих двух показателей  0.26, это говорит о слабой зависимости. 

# 12. Зависит ли **стоимость** игрока (Value) от его "**финтов**" (Skill Moves)?

# In[20]:


df['Value'] = df['Value'].replace("€", "", regex=True)


# In[21]:


def m_k(row):
        if 'M' in row:
            return float(row.replace("M", "")) * 1_000_000
        elif 'K' in row:
            return float(row.replace("K", "")) * 1_000


# In[22]:


df['Value'] = df['Value'].apply(m_k)


# In[23]:


df['Value']


# In[24]:


df.plot(x='Value', y='Skill Moves', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость между стоимостью игрока и его "финтов"' )
print('Коэффициент корреляции равен:', df['Value'].corr(df['Skill Moves']))


# Коэффициент корреляции этих двух показателей 0.31, это говорит о слабой зависимости, поэтому стоимость игрока (Value) немного зависит от его "финтов"

# 13. Покажите распределения основных характеристик игроков: **рост**, **вес**, **возраст** по **позициям** (Position), сделайте выводы

# In[25]:


df['Weight'] = df['Weight'].str.replace(r"[^\d\.]", "", regex=True).astype('float') #уберем lbs, чтобы привести к числовому


# In[26]:


df['Age'] = df['Age'].dropna()
df['Weight'] = df['Weight'].dropna() 


# In[27]:


group_labels = []
hist_data = []


# In[28]:


me = [i for i in df['Position'].unique() if i is not np.nan]


# In[29]:


for i in me:
    group_labels.append(i)
    qwe = df[df['Position']==i]
    hist_data.append(qwe['Age'])


# In[30]:


fig_age=ff.create_distplot(hist_data, group_labels, show_hist=False)
fig_age.update_layout(title_text='Возраст по позициям')
fig.layout.yaxis.update({'title': 'Возраст'})
fig_age.show()


# Возраст всех футболистов по позициям в основном от 20 до 27 лет. Самые молодые центральные полузащитники(20 лет), а самые старшие правые атакующие полузащитники. У остальных позиций чаще всего самый частый возраст 25. 

# In[31]:


df['Weight'] = df['Weight']*0.453592 #переведем вес в килограммы


# In[32]:


group_labels2 = []
hist_data2 = []


# In[33]:


for i in me:
    group_labels2.append(i)
    qwe = df[df['Position']==i]
    hist_data2.append(qwe['Weight'])


# In[34]:


fig_weight = ff.create_distplot(hist_data2, group_labels2, show_hist=False)
fig_weight.update_layout(title_text='Вес по позициям')
fig.layout.yaxis.update({'title': 'Вес'})
fig_weight.show()


# Вес футболистов в основном от 70 до 80. Больше всего с весом 70 кг. Самые тяжелые - центральные защитники и вратари, их вес чаще всего 78 кг. 

# In[35]:


df['Height']=df['Height'].str.replace("'", '.')
df['Height'] = df['Height'].astype('float64') #приведем значения роста к числовым 


# In[36]:


group_labels3 = []
hist_data3 = []


# In[37]:


for i in me:
    group_labels3.append(i)
    qwe = df[df['Position']==i]
    hist_data3.append(qwe['Height'])


# In[38]:


fig_height=ff.create_distplot(hist_data3, group_labels3, show_hist=False)
fig_height.update_layout(title_text='Рост по позициям')
fig.layout.yaxis.update({'title': 'Рост'})
fig_height.show()


# В основном все футболисты ростом 5'8,5'9(176 - 179 см), но вратари существенно выше остальных игроков, их рост чаще всего 6'2(188 см). Центральные защитники тоже выделяются своим ростом. 

# 14. Есть ли зависимость **скорости** (SprintSpeed) от **веса** и **роста**? От **национальности**?

# In[39]:


df.plot(x='SprintSpeed', y='Weight', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость скорости от веса')
print('Коэффициент корреляции равен:', df['SprintSpeed'].corr(df['Weight']))


# Коэффициент корреляции этих двух показателей -0.41, это говорит о средней зависимости. Если увеличится один параметр, то скорее всего уменьшится другой. 

# In[40]:


df.plot(x='SprintSpeed', y='Height', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость скорости от роста')
print('Коэффициент корреляции равен:', df['SprintSpeed'].corr(df['Height']))


# Коэффициент корреляции этих двух показателей -0.32, это говорит о слабой зависимости. 

# In[41]:


rar=(df
 .pivot_table(index='Nationality', values='SprintSpeed', aggfunc='mean')
 .sort_values('SprintSpeed', ascending = False)
 .reset_index()
 )
fig = px.line(rar, x='Nationality', y="SprintSpeed")
fig.update_layout(title_text='Зависимость скорости от национальности')
fig.show()


# In[42]:


len(df[df['Nationality'] =='Qatar'])


# In[43]:


len(df[df['Nationality'] =='Liberia'])


# Средняя скорость всех национальностей примерно одинаковая, футболисты из Либерии и Катара выделяются, но их количество слишком маленькое, поэтому делать вывод о зависимости скорости от национальности нельзя. Зависимость скороси от национальности отсутствует. 

# 15. Под какими номерами чаще всего играют нападающие? Визуализируйте частотность

# In[44]:


f = df[(df['Position'] == 'RF') | (df['Position'] == 'LF')]


# In[45]:


np.random.seed(1)
x = f['Jersey Number']
hist_data = [x]
group_labels = ['Номер']

fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text='Распределение номеров нападающих')
fig.layout.yaxis.update({'title': 'Номер'})
fig.show()


# Чаще всего нападающие играют под номером 10

# 16. Есть ли за зависимость **силы удара** (ShotPower) от **ведущей ноги**? от **агрессивности** (Aggression)? 
# 

# In[46]:


dar=(df
 .pivot_table(index='Preferred Foot', values='ShotPower', aggfunc='median')
 .sort_values('ShotPower', ascending = False)
 .reset_index()
 )
fig = px.line(dar, x='Preferred Foot', y="ShotPower")
fig.update_layout(title_text='Зависимость силы удара от ведущей ноги')
fig.show()


# Медианное значение силы удара правой и левой ноги отличаются не существенно. Сила удара левой ноги выше на 2. Зависимости нет. 

# In[47]:


df.plot(x='ShotPower', y='Aggression', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость силы удара  от агрессивности' )
print('Коэффициент корреляции равен:', df['ShotPower'].corr(df['Aggression']))


# Коэффициент корреляции этих двух показателей 0.49, это говорит о средней зависимости. Чем сильнее удар, тем агрессивнее футболист.  

# 17. Найдите суммарную стоимость игроков Value в каждом клубе
# 18. Визуализируйте соотношение сумарной стоимости топ 10 клубов

# In[48]:


val=(df
 .pivot_table(index='Club', values='Value', aggfunc='sum')
 .sort_values('Value', ascending = False)
 .reset_index()
 )
val


# In[49]:


fig = px.line(val.head(10), x='Club', y="Value", markers=True)
fig.update_layout(title_text='Соотношение сумарной стоимости топ 10 клубов')
fig.show()


# 19. Есть ли зависимость между продолжительностью контракта игрока и его стоимостью? 

# In[50]:


df['Joined'] = pd.to_datetime(
    df['Joined'], format='%b %d, %Y'
).dt.year


# In[51]:


df['Joined'] = df['Joined'].dropna().astype('int')


# In[52]:


def date(row):
    if row == '2021':
        return row
    elif row == '2022':
        return row
    elif row == '2020':
        return row
    elif row == '2023':
        return row
    elif row == '2019':
        return row
    elif row == '2024':
        return row
    elif row == '2025':
        return row
    elif row == '2026':
        return row
    elif row == '2018':
        return row
    else:
        return pd.to_datetime(row, format='%b %d, %Y').year


# In[53]:


df['Contract Valid Until'] = df['Contract Valid Until'].apply(date)


# In[54]:


df['Contract Valid Until'] = df['Contract Valid Until'].dropna().astype('float')


# In[55]:


df['long'] = df['Contract Valid Until'] - df['Joined']


# In[56]:


df.plot(x='Value', y='long', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость между продолжительностью контракта игрока и его стоимостью' )
print('Коэффициент корреляции равен:', df['Value'].corr(df['long']))


# Коэффициент корреляции этих двух показателей 0.23, это говорит о слабой зависимости.

# 20. В какой год будет больше всего свободных игроков на рынке (исходя из этих данных)? Визуализируйте зависимость

# In[57]:


unt=(df
 .pivot_table(index='Contract Valid Until', values='ID', aggfunc='count')
 .sort_values('Contract Valid Until', ascending = False)
 .reset_index()
 )
fig = px.line(unt, x='Contract Valid Until', y="ID", markers=True)
fig.update_layout(title_text='свободные игроки на рынке ')
fig.show()


# В 2019 году будет больше всего свободных игроков на рынке

# 21. Есть ли зависмость **потенциала** (Potential) игрока от **возраста**?
# 22. Всегда ли молодые игроки обладают большим потенциалом?

# In[58]:


df.plot(x='Potential', y='Age', kind='scatter', alpha=0.1, figsize=(15,8), title='зависимость потенциала игрока от возраста' )
print('Коэффициент корреляции равен:', df['Potential'].corr(df['Age']))


# Коэффициент корреляции этих двух показателей 0.25, это говорит о слабой зависимости.

# In[59]:


pot=(df
 .pivot_table(index='Age', values='Potential', aggfunc='mean')
 .sort_values('Potential', ascending = False)
 .reset_index()
 )
fig = px.line(pot, x='Potential', y="Age", markers=True)
fig.update_layout(title_text='потенциал по возрастам')
fig.show()


# Не только молодые игроки имеют самый большой потенциал. Самые перспективные футболисты, конечно, в возрасте 16-25, но футболисты в возрасте 45 лет также имеют достаточно высокий потенциал

# 23. Есть ли зависимость **рейтинга** игрока (Overall) от **возраста**?
# 24. Когда у игроков "пик" карьеры?
# 25. Игроков какого возраста больше всего с рейтингом 90+?

# In[60]:


print('Коэффициент корреляции равен:', df['Overall'].corr(df['Age']))


# Коэффициент корреляции этих двух показателей 0.25, это говорит о средней зависимости. Чем выше рейтинг, тем футболист старше. 

# In[61]:


fig = px.scatter(df, x="Age", y="Overall", marginal_x="histogram")
fig.update_layout(title_text='зависимость рейтинга игрока от возраста')
fig.show()


# Пик карьеры у футболистов в возрасте 21-26

# In[62]:


overall90 = df[df['Overall']>=90]


# In[63]:


(overall90
 .pivot_table(index='Age', values='Overall', aggfunc='count')
 .sort_values('Overall', ascending = False)
 .reset_index()
 )


# Игроков с рейтингом 90+ больше всего в возрасте 32 года. 

# 26. Насколько средняя зарплата (Wage) игроков с рейтингом (Overall) 90+ больше средней зарплаты с рейтингом [80 - 90]? Визуализируйте возможную зависимость

# In[64]:


df['Wage'].unique() # все зарплаты в тысячах, можем убрать


# In[65]:


df['Wage'] = df['Wage'].str.replace("[^\d\.]", "", regex=True).astype('int') 


# In[66]:


overall90 = df[df['Overall']>=90]


# In[67]:


a=overall90['Wage'].mean()


# In[68]:


overall80 = df[(df['Overall']<=90) & (df['Overall']>=80)]


# In[69]:


b=overall80['Wage'].mean()


# In[70]:


print('Средняя зарплата игроков с рейтингом 90+ больше средней зарплаты с рейтингом [80 - 90] на:  €', round(a-b),'K' )


# In[71]:


group_labels = ['90+', '80-90']

colors = ['slategray', 'magenta']
hist_data = [overall90['Wage'], overall80['Wage']]

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)


# Add title
fig.update_layout(title_text='Распределение зарплат среди футболистов с разными рейтингами')
fig.layout.yaxis.update({'title': 'Wage'})
fig.show()


# In[ ]:




