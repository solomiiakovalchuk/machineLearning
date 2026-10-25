# Джерело даних ЛР № 6

Набір: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Автори: W. Wolberg, W. Street, O. Mangasarian; ліцензія CC BY 4.0 (із зазначенням джерела).

Дані беруться з вбудованої функції `sklearn.datasets.load_breast_cancer`, тому файл у репозиторії не потрібен.
Розмір: 569 записів, 30 числових ознак (характеристики ядер клітин за цифровими зображеннями тонкоголкової аспірації, FNA);
212 злоякісних (malignant) та 357 доброякісних (benign). Позитивний клас задачі — malignant (`y = 1`, у sklearn це `target == 0`).
ID об'єкта — номер рядка в `load_breast_cancer`, з нуля. SHA-256 сирих даних, версія sklearn і примітка про попереднє використання
набору в інших лабораторних — у `../results/data_provenance.json`.
