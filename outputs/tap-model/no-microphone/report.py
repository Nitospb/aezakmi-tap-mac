from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
m=json.loads((R/'metrics.json').read_text());test=m['test_selected'];n=test['n']
def ci(p):
 z=1.96;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,c-h),min(1,c+h)]
m['test_confidence_intervals_95_wilson']={k:ci(test[k]) for k in ['exact','within_one','within_two','side_x','side_y']}
(R/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
labels={'imu':'Акселерометр + гироскоп','motion':'IMU + дробный угол','all':'IMU + дробный угол + свет'}
lines=['Обучение по новому сеансу, микрофон исключён','',f"Сеанс: {m['session']}",f"Действительных тапов: {m['dataset_valid']}; использовано: {m['used']}.",f"Автоматических: {m['automatic']}; ручных с оценённым временем: {m['manual_estimated']}.",f"Размер сетки: 20×13 = 260 клеток. Три прохода, по одному тапу на клетку в проходе.",'','Разделение по времени и проходам:',json.dumps(m['split'],ensure_ascii=False),'Первый проход — обучение; второй — выбор конфигурации; первые два — обучение перед тестом; третий — тест.','Для основного теста взяты только автоматически размеченные тапы. Ручные — отдельно.','',f"Выбрано на валидации: {m['selected']['config']} / {labels[m['selected']['group']]}",'Выбор: сначала доля точных клеток, затем доля в пределах одной клетки, затем средний промах.','Классификация X/Y, выбор максимального класса; усреднения координат нет.','',f'Результат на {n} отложенных автоматических тапах:']
for k,title in [('exact','Точная клетка'),('within_one','Не дальше 1 клетки по каждой оси'),('within_two','Не дальше 2 клеток по каждой оси'),('region_2x2','Правильная четверть экрана 2×2'),('region_3x3','Правильная область экрана 3×3'),('region_4x4','Правильная область экрана 4×4'),('side_x','Лево / право'),('side_y','Верх / низ')]:lines.append(f'{title}: {test[k]*100:.2f}% ({round(test[k]*n)}/{n})')
lines += [f"Средний промах: {test['mean_distance_cells']:.2f} клетки; медиана: {test['median_distance_cells']:.2f}.",f"Различных предсказанных клеток: {test['unique_predicted_cells']}.",'','Сравнение вариантов на одном и том же тесте:']
for g,v in m['test_by_feature_group'].items():lines.append(f"{labels[g]}: точно {100*v['exact']:.2f}%, рядом ≤1 {100*v['within_one']:.2f}%, средний промах {v['mean_distance_cells']:.2f}")
lines += ['', 'Нормализация всех моделей: z=(x−mean)/std. Параметры находятся в Pipeline и normalization.json.', 'Средние и std вычисляются заново на каждой обучающей части, без валидации или теста.', 'Отклонения сигналов от фона перед ударом также включены отдельными признаками.', '', 'Случайная равномерная точная клетка: 0,38%.','Пределы: тест — третий проход одного сеанса, не новый независимый сеанс.','Метки — подсвеченные клетки, фактическое положение пальца отдельно не измерялось.','В обучении ручные примеры имеют вес 0,3; время импульса оценено только по движению.','Модель model.joblib после оценки обучена заново на всех пригодных тапах.','Показанные цифры относятся к evaluation-*.joblib, обученным только на первых двух проходах.','Микрофон не использован ни в признаках, ни в выборе модели. Файлы аудио не открывались.','Старые приложения и модели сохранены. Новый model.joblib — Python-модель, отдельная от установленного TapPredictor.','Для воспроизведения: train.py, затем report.py в прежнем окружении work/model-env.']
(R/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
rows=json.loads((R/'heldout-predictions.json').read_text());truth=np.array([r['truth'] for r in rows]);pred=np.array([r['prediction'] for r in rows]);dist=np.linalg.norm(truth-pred,axis=1)
err=np.full((13,20),np.nan);counts=np.zeros((13,20))
for (x,y),d in zip(truth,dist):err[y,x]=d
for x,y in pred:counts[y,x]+=1
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
a=axs[0].imshow(err,origin='upper',vmin=0,vmax=max(5,float(np.nanpercentile(err,95))),cmap='magma');fig.colorbar(a,ax=axs[0],label='Euclidean error (grid cells)',shrink=.75);axs[0].set_title('Error at the instructed cell\nGray = no automatic test tap');axs[0].set_facecolor('#dddddd')
a=axs[1].imshow(counts,origin='upper',vmin=0,cmap='viridis');fig.colorbar(a,ax=axs[1],label='Predictions per cell',shrink=.75);axs[1].set_title('Where the model predicted')
for ax in axs:
 ax.set_xticks([0,4,9,14,19],['1','5','10','15','20']);ax.set_yticks([0,3,6,9,12],['1','4','7','10','13']);ax.set_xlabel('Column');ax.set_ylabel('Row')
fig.suptitle(f"No microphone | held-out pass 3: n={n} | exact {test['exact']*100:.1f}% | within 1 cell {test['within_one']*100:.1f}%")
fig.savefig(R/'heldout-map.png',dpi=160);plt.close(fig)
print('\n'.join(lines))
