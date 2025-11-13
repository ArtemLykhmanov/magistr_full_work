Project: ML-based Network Traffic Classification (reproducible thesis pipeline)
Опис проєкту

Даний проєкт присвячено розробці системи виявлення мережевих вторгнень (Intrusion Detection System, IDS) із використанням методів машинного навчання. Метою є побудова моделей, здатних класифікувати мережевий трафік як нормальний або зловмисний (різні типи атак) з високою точністю. Особливістю проєкту є випробування моделей на двох різних наборах даних – UNSW-NB15 та CICIDS2017 – і аналіз того, як модель, навчена на одному наборі, працює на іншому (перехресне тестування). Такий підхід імітує реальний сценарій, коли IDS-систему тренують на одних даних, а розгортають в іншому середовищі. В проєкті реалізовано кілька алгоритмів класифікації та проведено порівняння їх продуктивності, включно з оцінкою повноти виявлення атак, точності класифікації та площі під PR-кривою (PR-AUC).

Ключові можливості проєкту:

Підтримка двох популярних датасетів для IDS: UNSW-NB15 (2015 рік) та CICIDS2017 (2017 рік).

Попередня обробка даних: нормалізація ознак, узгодження схем даних обох датасетів для уможливлення між-наборового тестування.

Реалізація та навчання кількох моделей машинного навчання для задачі класифікації “атака/норма”.

Аналіз результатів: обчислення метрик (Accuracy, Precision, Recall, F1, PR-AUC), побудова кривих PR та F1 для порівняння моделей, визначення важливості ознак.

Можливість крос-доменного оцінювання: тестування моделі, навченої на одному наборі, на іншому для оцінки здатності до узагальнення на інше середовище.


Підготовка даних: Для обох наборів виконано очищення та підготовку: непотрібні поля (ідентифікатори, IP-адреси тощо) відкинуто, відсутні значення (якщо були) заповнено або вилучено. Оскільки схеми ознак UNSW-NB15 та CICIDS2017 різняться, для крос-доменного тестування був застосований підхід гармонізації ознак – виділено підмножину характеристик, спільних або аналогічних для обох наборів, щоб модель могла оперувати однаковими вхідними параметрами. Наприклад, такі ознаки, як тривалість сеансу, кількість вхідних/вихідних пакетів, байтів, частота пакетів, наявність прапорців (FIN/SYN/RST/...), присутні в тій чи іншій формі в обох наборах даних і були приведені до єдиного формату. Всі числові ознаки нормовані (стандартизовані) перед подачею в модель. Категоріальні поля (як-от протокол, стан з’єднання) були або закодовані числовими індексами, або розгорнуті методом one-hot encoding.

Моделі та методи

У проєкті реалізовано та протестовано кілька алгоритмів машинного навчання для класифікації мережевого трафіку:

Random Forest (RF) – ансамблевий метод на основі множини випадкових дерев рішень. Відомий своєю високою точністю і стійкістю до перенавчання, особливо на великих наборах ознак. У реалізації використано 100 дерев (або інше задане число) з критерієм Gini, глибина дерев не обмежена (якщо не зазначено інше). RF забезпечує також оцінку важливості ознак, що корисно для нашого аналізу.

XGBoost (XGB) – градієнтний бустинг на рішеннях деревах (реалізація XGBoost). Цей алгоритм часто показує найкращі результати в задачах класифікації завдяки поєднанню багатьох слабких моделей (дерев) в сильну, з оптимізацією за допомогою градієнтного спуску. Були використані стандартні параметри або підібрані емпірично, такі як кількість дерев, глибина та темп навчання.

Логістична регресія (LogReg) – базовий лінійний метод класифікації. В нашому випадку використовується як бенчмарк для порівняння зі складнішими нелінійними моделями.

Багатошаровий перцептрон (MLP) – нейронна мережа прямого поширення (повнозв’язний багатошаровий перцептрон). Містить вхідний шар, один або кілька прихованих шарів та вихідний шар з одним нейроном (для бінарної класифікації). В якості функції активації використовувалася ReLU для прихованих шарів і сигмоїда (або softmax) на виході. Навчання здійснювалося методом зворотного поширення помилки з оптимізатором (наприклад, Adam). Для запобігання перенавчанню могли застосовуватися регуляризація або dropout.

Histogram-based Gradient Boosting (HGB) – алгоритм бустингу, реалізований у sklearn (HistGradientBoostingClassifier), який ефективно працює на великих даних за рахунок дискретизації (бінінгу) ознак. Він схожий за ідеєю на XGBoost, але інтегрований у sklearn і добре оптимізований під числові дані.

Примітка: В коді передбачена можливість налаштування гіперпараметрів моделей. За замовчуванням використовувалися розумні стандартні параметри або налаштовані вручну. Для покращення підсумкової моделі рекомендовано провести повноцінний пошук гіперпараметрів (GridSearchCV, Bayesian optimization тощо), проте це виходить за рамки швидкої перевірки.

Data layout:
data/
UNSW_NB15/
UNSW_NB15/UNSW_NB15_training-set.csv
UNSW_NB15/UNSW_NB15_testing-set.csv
CICIDS2017/
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv


Outputs:
out/
tables/*.csv (ready-to-insert tables)
figures/*.png (ready-to-insert figures)
logs/*.json (run manifest)


Quickstart
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Quickstart (examples):

$models = "rf,hgb,mlp,xgb,rf_cal"
python experiments/eval_cross_many.py --scenario unsw_cv_test --target-fpr 0.01 --outdir out/unsw --models $models
CICIDS cross‑day (повний, усі пари, з аудитом) → out/cicids

$models = "rf,rf_cal,logreg,hgb,xgb"
python experiments/eval_cross_many.py --scenario cicids_cross_day --val monday --target-fpr 0.001 --outdir out/cicids --models $models
CICIDS cross‑day (2 дні, без аудиту, з калібруванням усіх моделей і “гладкими” reliability) → out/cicids_smooth

$models = "rf,rf_cal,logreg,hgb,xgb"
python experiments/eval_cross_many.py --scenario cicids_cross_day --val monday --target-fpr 0.001 --outdir out/cicids_smooth --models $models --pairs thursday_web,friday_ddos --disable-audit --calibrate-all
Cross‑dataset (CICIDS Tue+Wed → UNSW test) з узгодженим (harmonized) набором фіч → out/xdom_harm

$models = "rf,logreg,hgb,xgb,mlp"
python experiments/eval_cross_many.py --scenario cross_dataset --target-fpr 0.01 --outdir out/xdom_harm --models $models --align harmonized

Команди (для зручності тут усі команди що запускалися)

# 1) UNSW: CV on train + test (models as in current out/unsw)
python experiments/eval_cross_many.py --scenario unsw_cv_test --target-fpr 0.01 --outdir out/unsw --models rf,mlp,hgb,rf_cal,xgb

# 1b) UNSW repro-check run for RF only (writes to out/unsw_reprocheck)
python experiments/eval_cross_many.py --scenario unsw_cv_test --target-fpr 0.01 --outdir out/unsw_reprocheck --models rf

# 2) CICIDS cross-day without calibration (pairs as in current outputs)
python experiments/eval_cross_many.py --scenario cicids_cross_day --val monday --target-fpr 0.001 --outdir out/cicids_nocal --pairs thursday_web,friday_ddos --models rf,rf_cal,logreg,hgb,xgb

# 3) CICIDS cross-day with isotonic calibration for all models
python experiments/eval_cross_many.py --scenario cicids_cross_day --val monday --target-fpr 0.001 --outdir out/cicids_smooth --pairs thursday_web,friday_ddos --models rf,rf_cal,logreg,hgb,xgb --calibrate-all

# 4) Cross-dataset CIC(Tue+Wed)->UNSW(test), harmonized feature alignment
python experiments/eval_cross_many.py --scenario cross_dataset --align harmonized --target-fpr 0.01 --outdir out/xdom_harm --models rf,logreg,mlp,hgb,xgb

# 5) Train UNSW RF/XGB pipelines for McNemar test (saves to out/models)
python experiments/train_unsw_models_for_mcnemar.py --out-dir out/models

# 6) SHAP stability on UNSW (RF)
python experiments/shap_stability_unsw.py --n-splits 5 --sample-size 800 --out-dir out/unsw_shap --topk 10 20

# 7) Feature selection ablation (UNSW, RF)
python experiments/feature_selection_ablation.py --top-n 20 10

# 8) Figures used in write-up
# 8a) Baseline vs models (requires results/table_3_3_baseline_metrics.csv)
python experiments/plot_baseline_comparison.py --baseline-csv results/table_3_3_baseline_metrics.csv --unsw-test-csv out/unsw/tables/table_unsw_test.csv --cross-csv out/xdom_harm/tables/table_cross_dataset.csv --out out/fig_3_5_baseline_comparison.png

# 8b) F1 vs threshold: Dummy vs Logistic Regression on UNSW
python experiments/plot_f1_threshold_dummy_vs_model.py --dataset unsw --model logreg --out out/fig_3_6_f1_threshold_dummy_vs_logreg_unsw.png

Результати експериментів

На індивідуальних датасетах: Моделі показали дуже високі результати при навчанні і тестуванні в межах одного й того ж датасету. Зокрема, точність (Accuracy) досягала ~97-99%, а значення F1 та PR-AUC – близькі до 1.0, для обох наборів даних. Наприклад, Random Forest на UNSW-NB15 в наших експериментах дає точність близько 98%, а PR-AUC понад 0.99, що свідчить про відмінну здатність виявляти атаки, коли розподіл даних тренування і тесту однаковий. Моделі XGBoost та HGB також досягають порівнянної або дещо вищої ефективності. Простішій логістичній регресії, очікувано, трохи не вистачає нелінійності для максимальних показників, але вона теж демонструє високий результат на цих даних (близько 95% точності). Нейронна мережа (MLP) після налаштування параметрів (наприклад, вибору оптимальної кількості нейронів у прихованому шарі) також виходить на рівень ~96-98% точності.

Перехресне тестування між наборами: Коли модель, навчена на UNSW-NB15, перевіряється на CICIDS2017 або навпаки, спостерігається значне падіння ефективності. Це зумовлено різницею у розподілах даних і особливостях ознак двох наборів. Наприклад, якщо навчити модель на UNSW-NB15 і застосувати до CICIDS2017, значення PR-AUC можуть знизитися до ~0.70-0.80 (замість >0.95 на вихідному наборі), а точність – впасти до ~60-70%, залежно від моделі. Аналогічно і у зворотному напрямі. Найкраще у цьому сценарії проявляють себе ансамблеві методи (Random Forest, XGBoost) – вони дещо стійкіші до зміни датасету, ніж лінійні моделі. Проте загалом крос-доменні результати підтверджують очікуване: без додаткових заходів з адаптації моделі до нового домену її здатність виявляти атаки помітно погіршується. Це важливий висновок роботи, який демонструє необхідність методів transfer learning або додаткового донавчання на новому середовищі, якщо планується розгортання IDS на інших даних.

Аналіз ознак: Для моделей Random Forest, XGBoost та HGB були проаналізовані ваги/важливості ознак. Виявилося, що для обох датасетів ключову роль грають схожі типи ознак: наприклад, сумарна кількість байтів і пакетів, тривалість сеансу, середня кількість пакетів за секунду, кількість з’єднань з певним станом чи поєднанням портів. Було складено рейтинг найбільш інформативних полів; використання лише топ-10 із них майже не знижує точність (в деяких випадках UNSW-NB15 все ще дає ~98.5% з 12 кращими ознаками). Це говорить про те, що дані містять значну надлишковість, і можлива подальша оптимізація шляхом відбору ознак. У проєкті передбачено можливість запускати модель на відібраному піднаборі ознак (параметр --features), що може пришвидшити роботу і зменшити складність моделі без втрати якості.