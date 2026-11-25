from pathlib import Path
from io import BytesIO
import streamlit as st
import pandas as pd
from core import load_data, analyze, forecast, FEATURES
st.set_page_config(page_title='UniMonitor AI',layout='wide')
st.title('UniMonitor AI | Мониторинг вузов')
import os
import sqlite3
from storage import Store
store=Store(Path(os.getenv('UNIMONITOR_DB','unimonitor.sqlite')))
initial=os.getenv('UNIMONITOR_ADMIN_PASSWORD')
if initial:
    store.bootstrap(os.getenv('UNIMONITOR_ADMIN_LOGIN','admin'),initial)
if 'login' not in st.session_state:
    with st.form('login_form'):
        login=st.text_input('Логин')
        password=st.text_input('Пароль',type='password')
        if st.form_submit_button('Войти'):
            role=store.authenticate(login,password)
            if role:
                st.session_state.login=login
                st.session_state.role=role
                st.rerun()
            else: st.error('Неверный логин или пароль')
    st.caption('Первый администратор создаётся через переменные окружения (см. README).')
    st.stop()
actor=st.session_state.login
role=st.session_state.role
st.sidebar.caption(f'{actor} | {role}')
if st.sidebar.button('Выйти'):
    st.session_state.clear(); st.rerun()
if role=='admin':
    with st.sidebar.expander('Управление пользователями'):
        with st.form('new_user'):
            new_login=st.text_input('Новый логин')
            new_password=st.text_input('Новый пароль',type='password')
            new_role=st.selectbox('Роль',['viewer','analyst','admin'])
            if st.form_submit_button('Создать'):
                try: store.create_user(actor,new_login,new_password,new_role); st.success('Пользователь создан')
                except (ValueError,sqlite3.IntegrityError) as e: st.error(str(e))
    with st.sidebar.expander('Журнал действий'):
        st.dataframe(store.audit(actor))
if role in ('admin','analyst'):
    with st.sidebar.expander('Импорт набора'):
        with st.form('import_form'):
            file=st.file_uploader('CSV',type=['csv'])
            name=st.text_input('Название набора')
            origin=st.text_input('Источник и единицы измерения')
            kind=st.selectbox('Тип данных',['external','demo'],format_func=lambda v: 'Внешние (источник указан пользователем)' if v=='external' else 'Демонстрационные')
            if st.form_submit_button('Сохранить'):
                try:
                    if file is None: raise ValueError('Выберите файл')
                    store.save(actor,name,origin,kind,file.getvalue().decode('utf-8-sig'))
                    st.success('Набор сохранён')
                except (ValueError,UnicodeError) as e: st.error(str(e))
records=store.list(actor)
choice=st.sidebar.selectbox('Набор данных',[0]+[r['id'] for r in records],format_func=lambda i: 'Учебный демонстрационный набор' if i==0 else next(r['name'] for r in records if r['id']==i))
if choice==0:
    st.warning('Синтетические данные. Не официальная статистика.')
    source=Path(__file__).parent/'data/demo.csv'
else:
    record=store.read(actor,choice)
    st.caption('Источник: '+record['source'])
    st.warning('Демонстрационные данные' if record['kind']=='demo' else 'Внешний набор: достоверность источника не проверена системой.')
    source=BytesIO(record['csv'].encode())
try: df=load_data(source)
except ValueError as e: st.error(str(e)); st.stop()
st.sidebar.download_button('Шаблон CSV',pd.DataFrame(columns=['university','region','year']+FEATURES).to_csv(index=False),'template.csv')
labels=dict(zip(FEATURES,['Численность студентов','Средний балл','Научная деятельность','Доход','Трудоустройство, %']))
page=st.sidebar.radio('Раздел',['Обзор','Сравнение','Кластеры и аномалии','Прогноз','Эксперимент моделей','Исследование данных','Отчёты','Данные'])
if page=='Обзор':
    a,b,c=st.columns(3); a.metric('Вузов',df.university.nunique()); b.metric('Наблюдений',len(df)); c.metric('Последний год',int(df.year.max()))
    st.line_chart(df.groupby('year').employment.mean(),x_label='Год',y_label='Трудоустройство, %')
elif page=='Сравнение':
    year=st.selectbox('Год',sorted(df.year.unique(),reverse=True)); col=st.selectbox('Показатель',FEATURES,format_func=lambda x:labels[x]); names=st.multiselect('Вузы',sorted(df.university.unique()),default=sorted(df.university.unique())[:5]); subset=df[(df.year==year)&df.university.isin(names)]; st.bar_chart(subset.set_index('university')[col]); st.dataframe(subset,use_container_width=True)
elif page=='Кластеры и аномалии':
    try:
        result=analyze(df); st.info('Аномалия — верхние 10% ошибок реконструкции; требует экспертной проверки.'); st.dataframe(result,use_container_width=True); st.download_button('Экспорт анализа',result.to_csv(index=False),'analysis.csv')
    except ValueError as e: st.error(str(e))
elif page=='Прогноз':
    try:
        metrics,result=forecast(df); st.caption('Контрольная выборка — последний доступный целевой год.'); st.dataframe(pd.DataFrame(metrics).T); st.dataframe(result,use_container_width=True); st.download_button('Экспорт прогноза',result.to_csv(index=False),'forecast.csv')
    except ValueError as e: st.error(str(e))
elif page=='Эксперимент моделей':
    from research import experiment
    st.caption('Подбор параметров на прошлых годах; последний целевой год исключён из подбора.')
    if st.button('Выполнить эксперимент'):
        try:
            with st.spinner('Обучение и временная валидация'):
                result=experiment(df)
            st.write('Выбранная по валидации модель:',result['winner'])
            st.dataframe(result['ranking'])
            st.write('Отложенный год:',result['test_year'])
            st.dataframe(result['comparison'])
            st.bar_chart(result['comparison'].set_index('model')['MAE'])
            st.dataframe(result['forecast'])
            st.download_button('Экспорт эксперимента',result['comparison'].to_csv(index=False),'comparison.csv')
        except ValueError as e: st.error(str(e))
elif page=='Исследование данных':
    from eda import explore
    result=explore(df)
    st.subheader('Охват наблюдений')
    st.bar_chart(result['coverage'].set_index('year')['universities'])
    st.subheader('Описательная статистика')
    st.dataframe(result['summary'])
    st.subheader('Корреляции Спирмена за последний год')
    st.dataframe(result['correlation'].rename(index=labels,columns=labels))
    st.caption('Корреляция не означает причинную связь.')
    st.subheader('Годовые изменения')
    st.dataframe(result['changes'])
    st.download_button('Экспорт изменений',result['changes'].to_csv(index=False),'changes.csv')
elif page=='Отчёты':
    from reporting import build_report
    from research import experiment
    st.subheader('Формирование аналитического отчёта')
    include = st.checkbox('Включить повторное обучение и контрольные прогнозы')
    if st.button('Сформировать отчёт'):
        try:
            with st.spinner('Формирование отчёта'):
                result = experiment(df) if include else None
                report = build_report(df, 'Синтетический учебный набор' if choice==0 else record['source'],
                                      'demo' if choice==0 else record['kind'], result)
            st.session_state.report_bundle = (choice, include, report)
        except ValueError as e: st.error(str(e))
    saved = st.session_state.get('report_bundle')
    if saved and saved[:2] == (choice, include):
        html, bundle, manifest = saved[2]
        st.json(manifest)
        st.download_button('Скачать HTML', html, 'report.html', mime='text/html')
        st.download_button('Скачать полный архив отчёта', bundle, 'report.zip', mime='application/zip')
else:
    st.dataframe(df,use_container_width=True); st.download_button('Экспорт данных',df.to_csv(index=False),'data.csv')
