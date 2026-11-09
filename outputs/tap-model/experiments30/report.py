from pathlib import Path
import re,json,numpy as np,joblib
import common,exp23
ROOT=Path(__file__).resolve().parent
z=common.load();report=json.loads((ROOT/'metrics.json').read_text());registry=[]
allresults=[]
for f in sorted(ROOT.glob('*-result.json')):
 r=json.loads(f.read_text())
 if re.match(r'exp\d\d',f.name):allresults.append(r)
for i in range(1,31):
 agent=f'exp{i:02d}';source=ROOT/f'{agent}.py';assert source.exists(),source
 rows=[r for r in allresults if re.match(agent+r'[-_]',r.get('name',''))]
 entry=dict(agent=agent,source=source.name,result_count=len(rows),results=[r['name'] for r in rows])
 if rows:entry['best']=max(rows,key=lambda r:r['validation']['score'])
 else:assert i==28;entry['summary']='exp28-summary.json';entry['kind']='confidence/rejection rules; not counted as improved recognition'
 registry.append(entry)
(ROOT/'experiments.json').write_text(json.dumps(dict(agent_count=30,predictive_variants=len(allresults),confidence_rules=2,experiments=registry),ensure_ascii=False,indent=2))
# Reproduce selected evaluation model and verify all scaler fit scopes.
mask=(z['idx']>=520)&(z['w']==1)
for name,expected in [('evaluation-model.joblib',int((z['idx']<520).sum())),('model.joblib',len(z['idx']))]:
 model=joblib.load(ROOT/name)
 for h in model['heads']:
  scaler=h.named_steps['standardscaler'];assert scaler.n_samples_seen_==expected
  X=np.column_stack([z['M'],z['F']]);assert np.isfinite(scaler.transform(X)).all()
 if name=='evaluation-model.joblib':
  fine=[(1-model['base']['weight'])*a+model['base']['weight']*b for a,b in zip(common.probabilities(model['base']['motion'],z['M'][mask]),common.probabilities(model['base']['light'],z['F'][mask]))]
  sides=common.probabilities(model['heads'],X[mask],classes=(2,2));p=exp23.guard_probabilities(fine,sides,model['threshold']);pred=common.decode(p);expect=np.array([r['prediction'] for r in json.loads((ROOT/'heldout-predictions.json').read_text())]);assert np.array_equal(pred,expect)
report['verification']='30 agent source modules and per-agent results present; saved guard reproduces every one of 217 historical test predictions; scaler counts 519 evaluation/779 final; audio excluded.';report['app_deployment']='Current app kept unchanged: improvement on historical test too small to justify replacing it.';(ROOT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
base=report['test_baseline'];new=report['test_guard'];n=base['n']
lines=['30 независимых агентских экспериментов без микрофона','',f"Выполнено агентов: 30; вариантов распознавания: {len(allresults)}; дополнительно два правила уверенности (агент28).",'Одновременно работали не более трёх вспомогательных агентов.','Данные: 779 пригодных тапов из нового сеанса, сетка20×13, три прохода.','Все агенты выбирали варианты только по первому/второму проходам; третий не использовался ими для выбора.','Третий проход уже оценивался в прежних итерациях: это повторное сравнение, не независимый новый сеанс.','', 'Проверенные направления:','- окна80/130/300/400мс, ранняя и поздняя часть, адаптивный конец и выравнивание по началу/пику;','- шумовая нормализация, проекция по гравитации, интегралы/корреляции/спектры/затухание/вейвлеты;','- отбор признаков, PCA, шаблоны со сдвигом времени, LDA, SVM, ExtraTrees, RandomForest, LightGBM;','- штраф за дальние классы, дискретный риск, порядковые границы, иерархия2×2/4×4, веса краёв;','- независимые классификаторы сторон, качество ручных меток, ансамбли, диагностика уверенности.','', 'Окна восстановлены из непрерывного samples.bin:650мс до и500мс после удара.','Каждое автоматическое окно заканчивается до зелёного подтверждения; минимальная задержка подтверждения560мс.','Интерполяция на общую сетку не повышает аппаратную частоту измерений.','', 'Лучший по проверочным проходам вариант: exp23-guard80.','Если независимый классификатор стороны уверенно противоречит мелкой клетке, запрещается только противоположная половина, а выбирается максимальный класс внутри разрешённой. Крайние клетки сохраняются; координаты не усредняются.','На прямом проверочном проходе: точно10,6→11,9%, рядом54,9→57,1%, полэкрана6,6→4,9%.','На обратном разделении: точно10,1→10,1%, рядом45,4→45,8%, полэкрана7,6→5,9%.','',f'Итоговое повторное сравнение на {n} автоматических тапах третьего прохода:']
for key,title in [('exact','Точная клетка'),('within_one','Не дальше одной клетки по каждой оси'),('within_two','Не дальше двух клеток по каждой оси'),('far_half_screen','Промах на полэкрана'),('side_x','Правильно лево/право'),('side_y','Правильно верх/низ')]:
 lines.append(f"{title}: {100*base[key]:.2f}% ({round(n*base[key])}/{n}) → {100*new[key]:.2f}% ({round(n*new[key])}/{n}).")
lines += [f"Средний промах: {base['mean_distance_cells']:.3f} → {new['mean_distance_cells']:.3f} клетки.",'Полэкрана: ошибка≥10 столбцов или≥7 строк.','Новая модель изменила12 ответов, но не добавила и не потеряла точных попаданий.','Существенного устойчивого выигрыша не получено. Текущая модель приложения оставлена без замены.','', 'Отдельный практический результат: при отказе от точной точки в половине самых неуверенных случаев можно отсечь большую часть дальних ошибок на валидации. Это снижение покрытия, не улучшение точности на всех тапах.','Просто расширенные окна/сильный штраф/веса краёв не дали общего улучшения в этих проверках; это не доказательство бесполезности других реализаций.','', 'Артефакты:','experiments.json — все30 агентов и варианты; exp01.py..exp30.py — воспроизводимый код.','model.joblib — отдельная финальная экспериментальная модель, обученная после оценки на всех779 записях.','evaluation-model.joblib — модель первых двух проходов, которой соответствуют проценты.','selection.json/reverse-confirmation.json — выбор без третьего прохода и проверка устойчивости.','heldout-predictions.json — сравнение каждого из217 примеров.','', 'Лучшие варианты на валидации (не путать с итоговым тестом):']
for r in sorted(allresults,key=lambda r:r['validation']['score'],reverse=True)[:12]:
 v=r['validation'];lines.append(f"{r['name']}: score{v['score']:.4f}, точно{100*v['exact']:.2f}%, рядом{100*v['within_one']:.2f}%, полэкрана{100*v['far_half_screen']:.2f}%.")
(ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('Verified all 30 experiment sources; predictive variants:',len(allresults));print('Guard saved-model parity passed; app unchanged.');print('Test exact',new['exact'],'near',new['within_one'],'far',new['far_half_screen'])
