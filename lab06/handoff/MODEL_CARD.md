# Model card — Digits, digit 5 vs rest

Навчальний прототип маршрутизації зображень 8×8 для оператора OCR. Один об'єкт — 64 пікселі 0–16.
Позитивний клас: цифра 5; походження: sklearn load_digits / UCI Optical Recognition of Handwritten Digits.
Джерело: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
Ліцензія першоджерела UCI: CC BY 4.0; див. https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Вибрана гілка: manual; сімейство: lrl2.
Причина: Mean AP difference within decision tie region; choose smaller serialized pipeline. This is not proof of equivalence.
Рішення SHA-256: 27112edb474280b211723ae3621c9ac31b738d00d75e5867788ef9efe118f382. Test AP=0.994419, F1=0.944444, recall=0.944444.
Поріг 0.5 зафіксований до оцінювання. Pipeline навчається на train, не перевчався на test.
CV, параметри, схема, середовище і приклад інференсу — у цьому пакеті та ../results/.

Обмеження: мала історична колекція, немає ID авторів для group split; нові почерки та сканери не перевірені.
Це не розпізнавач усіх цифр. Не використовувати для автоматичного ухвалення фінансових чи інших значущих рішень.
За невідомої схеми, виходу пікселів за діапазон або дрейфу — зупинити інференс і переглянути модель з оператором.
До ЛР7: перевіряти схему, версії, хеш рішення, контрольний прогноз; перед реальним застосуванням потрібен незалежний набір авторів.
