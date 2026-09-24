# task_modul_5

Predictia pretului masinilor second-hand

1. Descrierea proiectului.

Scopul acestui proiect este dezvoltarea unui model de Machine Learning de regresie care estimeaza pretul unei masini second-hand pe baza caracteristicilor acesteia.

Variabila tinta este `priceUSD`, iar predictia este realizata folosind informatii precum marca, modelul, vechimea masinii, kilometrajul, starea vehiculului, tipul de combustibil, capacitatea cilindrica, transmisia, tractiunea si segmentul masinii.

Proiectul urmareste intregul flux de lucru al unui proiect de Machine Learning:

EDA -> Curatarea datelor -> Ingineria caracteristicilor -> Preprocesarea datelor -> Antrenarea modelului -> Evaluarea modelului -> Compararea modelelor -> Selectarea modelului final.



2. Dataset.

Dataset-ul initial contine 56.244 de observatii si 12 coloane.

Coloanele sunt:

- `make` - marca masinii
- `model` - modelul masinii
- `priceUSD` - pretul masinii in USD si variabila tinta
- `year` - anul fabricatiei
- `condition` - starea masinii
- `mileage(kilometers)` - kilometrajul
- `fuel_type` - tipul de combustibil
- `volume(cm3)` - capacitatea cilindrica
- `color` - culoarea
- `transmission` - tipul transmisiei
- `drive_unit` - tipul tractiunii
- `segment` - segmentul masinii


3.EDA-Analiza exploratorie.

Analiza exploratorie a fost realizata in notebook-ul Jupyter dedicat EDA.

Au fost analizate:

- dimensiunea dataset-ului;
- tipurile de date;
- valorile lipsa;
- distributia variabilei tinta `priceUSD`;
- coloanele numerice;
- coloanele categoriale;
- valorile extreme;
- relatia dintre pret si caracteristicile numerice;
- relatia dintre pret si caracteristicile categoriale;
- randurile duplicate.


3.1. Variabila tinta.

Pretul mediu al unei masini este de aproximativ 7.415 USD, iar mediana este de aproximativ 5.350 USD.

Diferenta dintre medie si mediana, impreuna cu valorile extreme observate, arata ca distributia pretului este asimetrica spre dreapta.

Preturile variaza intre 48 USD si 235.235 USD.

Analiza masinilor cu preturi foarte mici si foarte mari a aratat ca valorile extreme nu sunt automat erori. Preturile foarte mici apar inclusiv pentru masini vandute pentru piese sau masini foarte vechi, iar preturile foarte mari apar pentru modele premium si de lux.

Din acest motiv, preturile extreme nu au fost eliminate automat.


3.2. Coloanele numerice.

Principalele coloane numerice analizate au fost:

- `priceUSD`
- `year`
- `mileage(kilometers)`
- `volume(cm3)`

Anul fabricatiei variaza intre 1910 si 2019.

Au fost identificate valori foarte mari pentru kilometraj, inclusiv valori de cateva milioane de kilometri. Unele dintre acestea aveau tipare precum 9.999.999, 8.888.888, 7.777.777 sau 6.666.666 km, ceea ce indica valori introduse incorect sau valori de tip placeholder.

Au fost observate si capacitati cilindrice foarte mari. Analiza masinii asociate fiecarei valori a aratat ca unele motoare mari sunt plauzibile, in timp ce altele nu corespund realist modelului masinii.


3.3. Coloanele categoriale.

Dataset-ul contine coloane categoriale cu niveluri diferite de cardinalitate.

`make` contine 96 de marci, iar `model` contine 1.034 de valori unice. `model` este astfel o caracteristica cu cardinalitate ridicata.

Alte observatii importante:

- `condition` contine 3 categorii si este dominata de `with mileage`;
- `fuel_type` contine petrol, diesel si electrocar, masinile electrice fiind foarte rare;
- `color` contine 13 categorii;
- `transmission` contine doua categorii;
- `drive_unit` contine valori lipsa;
- `segment` contine de asemenea un numar important de valori lipsa.

Analiza preturilor pe categorii a aratat ca marca, modelul, starea masinii, transmisia, tractiunea si segmentul contin informatii relevante pentru estimarea pretului.


3.4. Duplicate.

In dataset au fost identificate 87 de randuri duplicate.

Acestea au fost eliminate in etapa de curatare pentru a evita reprezentarea repetata a unor observatii identice.


4. Curatarea datelor.

Principalele operatii sunt:

1. standardizarea denumirilor coloanelor;
2. eliminarea spatiilor inutile din valorile textuale;
3. standardizarea valorilor lipsa;
4. standardizarea valorilor categoriale;
5. conversia coloanelor numerice;
6. tratarea kilometrajelor nerealiste;
7. tratarea capacitatilor cilindrice nerealiste;
8. eliminarea randurilor duplicate;
9. eliminarea observatiilor fara valoarea tinta.


4.1. Kilometraj.

In urma EDA au fost observate numeroase valori de peste 1.000.000 km, inclusiv valori repetitive si foarte putin plauzibile.

Valorile:

`mileage_kilometers > 1.000.000`

sau valorile negative sunt considerate invalide.

In loc sa fie eliminat intregul rand, kilometrajul respectiv este transformat intr-o valoare lipsa. Astfel sunt pastrate celelalte informatii valide despre masina.

Dupa curatare:

- nu mai exista kilometraje peste 1.000.000 km;
- nu exista kilometraje negative;
- 362 de valori ale kilometrajului sunt lipsa.


4.2. Capacitatea cilindrica

EDA a aratat ca motoarele de capacitate mare nu trebuie eliminate automat. De exemplu, valori de aproximativ 6.000 - 7.000 cm3 pot fi valide pentru anumite masini.

In acelasi timp, au fost identificate valori foarte mari care nu corespundeau realist modelelor respective.

Din acest motiv, au fost considerate invalide valorile:

`volume_cm3 < 500`

sau:

`volume_cm3 > 7000`

Aceste valori sunt transformate in valori lipsa, fara eliminarea intregii observatii.

Dupa curatare, capacitatea cilindrica maxima este de 7.000 cm3.


4.3. Valorile lipsa

Dupa curatare, principalele valori lipsa sunt:

- `mileage_kilometers`: 362
- `volume_cm3`: 141
- `drive_unit`: 1.904
- `segment`: 5.285

Aceste observatii nu au fost eliminate, deoarece celelalte caracteristici ale masinii pot ramane utile pentru model.

Valorile lipsa sunt tratate ulterior in pipeline-ul de preprocessing.


5. Ingineria caracteristicilor.

Au fost create doua caracteristici noi.

5.1. `car_age`

`car_age` reprezinta varsta masinii si este calculata folosind anul 2019 ca an de referinta:

`car_age = 2019 - year`

Anul 2019 a fost ales deoarece reprezinta anul maxim din dataset.

Aceasta caracteristica exprima mai intuitiv vechimea unei masini decat anul de fabricatie.

5.2. `mileage_per_year`

A fost creata caracteristica:

`mileage_per_year = mileage_kilometers / car_age`

Aceasta reprezinta kilometrajul mediu raportat la vechimea masinii si ofera o informatie despre intensitatea utilizarii vehiculului.

Pentru masinile cu `car_age = 0`, varsta utilizata in calcul este 1 pentru a evita impartirea la zero.

In dataset, mediana acestei caracteristici este de aproximativ 16.154 km/an.


6. Data Preprocessing.

Variabila tinta este:

`priceusd`

Caracteristicile numerice utilizate de model sunt:

- `mileage_kilometers`
- `volume_cm3`
- `car_age`
- `mileage_per_year`

Caracteristicile categoriale sunt:

- `make`
- `model`
- `condition`
- `fuel_type`
- `color`
- `transmission`
- `drive_unit`
- `segment`

Coloana `year` nu este folosita impreuna cu `car_age`, deoarece cele doua coloane contin aceeasi informatie exprimata in mod diferit.


6.1. Preprocesarea coloanelor numerice.

Pentru coloanele numerice este utilizat un pipeline format din:

`SimpleImputer(strategy="median")`

si:

`StandardScaler()`

Valorile lipsa sunt completate folosind mediana. Mediana a fost aleasa deoarece este mai putin influentata de valorile extreme decat media.

`StandardScaler` standardizeaza caracteristicile numerice astfel incat acestea sa fie reprezentate pe scari comparabile.


6.2. Preprocesarea coloanelor categoriale.

Pentru coloanele categoriale sunt utilizate:

`SimpleImputer(strategy="most_frequent")`

si:

`OneHotEncoder(handle_unknown="ignore")`

Valorile lipsa din coloane precum `drive_unit` si `segment` sunt completate cu cea mai frecventa categorie.

One-Hot Encoding transforma categoriile in reprezentari numerice fara a introduce o ordine artificiala intre acestea.

Optiunea `handle_unknown="ignore"` permite pipeline-ului sa proceseze categorii care nu au fost intalnite in setul de antrenare.

Toate transformarile sunt combinate folosind `ColumnTransformer`.


7. Impartirea datelor.

Datele au fost impartite in:

- 80% set de antrenare;
- 20% set de testare.

A fost utilizat:

`random_state=42`

pentru reproductibilitatea rezultatelor.

Au rezultat:

- 44.924 observatii pentru antrenare;
- 11.232 observatii pentru testare.

Toate modelele comparate au utilizat aceeasi impartire train/test.


8. Model baseline - Linear Regression.

Primul model antrenat a fost `LinearRegression`.

Acesta a fost folosit ca model de baza, deoarece este un algoritm simplu si ofera un punct de referinta pentru evaluarea modelelor mai complexe.

Rezultatele obtinute au fost:

| Metrica | Valoare |
|---|---:|
| MAE | 2.031,72 USD |
| MSE | 15.509.300 |
| RMSE | 3.938,19 USD |
| R2 | 0,7483 |

MAE arata ca modelul greseste in medie cu aproximativ 2.032 USD.

R2 de aproximativ 0,748 arata ca modelul explica o parte importanta din variatia preturilor, dar analiza erorilor a evidentiat si predictii foarte slabe pentru anumite masini.

De asemenea, Linear Regression a generat in unele cazuri preturi negative, care nu au sens in contextul problemei.


9. Compararea modelelor.

Au fost comparate patru modele de regresie:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Pentru o comparatie corecta, toate modelele au utilizat acelasi preprocessing si aceeasi impartire train/test.

Rezultatele obtinute au fost:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Random Forest | **1.052,02** | **2.404,99** | **0,9061** |
| Decision Tree | 1.356,70 | 3.149,41 | 0,8390 |
| Gradient Boosting | 1.537,67 | 2.926,54 | 0,8610 |
| Linear Regression | 2.031,72 | 3.938,19 | 0,7483 |

Random Forest a obtinut cel mai mic MAE si RMSE si cel mai mare R2 dintre modelele testate.


10. Alegerea modelului final.

Modelul ales este:

**Random Forest Regressor**

Acesta a obtinut:

- MAE: aproximativ **1.052 USD**
- RMSE: aproximativ **2.405 USD**
- R2: aproximativ **0,906**

Comparativ cu Linear Regression, eroarea absoluta medie a scazut de la aproximativ 2.032 USD la 1.052 USD, ceea ce reprezinta o reducere de aproximativ 48%.

Random Forest este potrivit pentru aceasta problema deoarece poate surprinde relatii neliniare si interactiuni intre caracteristici.

Pretul unei masini nu depinde liniar de o singura caracteristica, ci de combinatii intre marca, model, vechime, kilometraj, stare, capacitatea motorului si alte caracteristici.

Din aceste motive, Random Forest a fost ales ca model final.


11. Interpretarea modelului final.

Random Forest a obtinut rezultate considerabil mai bune decat modelul LinearRegression.

Un MAE de aproximativ 1.052 USD inseamna ca, pe setul de test, diferenta absoluta dintre pretul real si cel estimat este in medie de aproximativ 1.052 USD.

R2 de aproximativ 0,906 indica faptul ca modelul explica aproximativ 90,6% din variatia preturilor din setul de test.

Totusi, modelul nu prezice toate masinile cu aceeasi precizie.

Cele mai mari erori apar in special pentru unele masini cu preturi foarte ridicate sau caracteristici neobisnuite. De exemplu, pentru o masina cu pretul real de 120.000 USD, modelul a estimat aproximativ 41.908 USD.

Diferenta dintre MAE si RMSE indica de asemenea existenta unui numar redus de predictii cu erori foarte mari.

Prin urmare, modelul ofera rezultate bune la nivel general, dar predictiile pentru masini rare, foarte scumpe sau neobisnuite trebuie interpretate cu prudenta.


12. Structura proiectului este:

|-- cars.csv

|-- cars_cleaned.csv

|-- cars_cleaned_with_features.csv

|    

|-- EDA.ipynb

|

|-- data_cleaning.py

|-- build_features.py

|-- data_preprocessing.py

|

|-- train_model.py

|-- test_model.py

|-- evaluate_model.py

|
    
|-- model_comparison.py

|
    
|-- train_model2.py

|-- test_model2.py

|-- evaluate_model2.py

|

|-- linear_regression_model.joblib

|-- random_forest_model.joblib

|

|-- Readme.md
    


13. Concluzie.

Proiectul a urmarit intregul proces de dezvoltare a unui model de regresie pentru estimarea pretului masinilor second-hand.

EDA a permis identificarea valorilor lipsa, a valorilor extreme, a duplicatelor si a relatiilor dintre caracteristicile masinii si pret.

In etapa de curatare au fost tratate valorile nerealiste fara eliminarea inutila a observatiilor care contineau alte informatii valide.

Au fost create caracteristicile `car_age` si `mileage_per_year`, iar datele numerice si categoriale au fost pregatite printr-un pipeline de preprocessing.

Au fost comparate patru modele de regresie. Random Forest a obtinut cele mai bune rezultate, cu un MAE de aproximativ 1.052 USD si un R2 de aproximativ 0,906, fiind ales drept model final.

Rezultatele arata ca modelul poate estima cu o precizie buna pretul majoritatii masinilor din dataset, dar exista in continuare erori importante pentru unele masini rare sau foarte scumpe.
