% ------------------------------------ Факты ----------------------------------

% Хранение задач с числовыми характеристиками
% задача(Идентификатор, TrendStrength, SeasonalityStrength, RankedMethods).
задача(electricity, 0.81,  0.94,  [patch_tst, crossformer, dlinear]).
задача(metr-la,     0.269, 0.521, [nlinear, triformer, patch_tst]).
задача(pems-bay,    0.145, 0.629, [crossformer, patch_tst, dlinear]).
задача(pems04,      0.339, 0.917, [crossformer, times_net, patch_tst]).
задача(pems08,      0.376, 0.913, [crossformer, patch_tst, times_net]).
задача(solar, 		0.478, 0.919, [crossformer, patch_tst, micn]).


% функция расстояния (евклидова)
расстояние(X1, Y1, X2, Y2, D) :-
    DX is X1 - X2,
    DY is Y1 - Y2,
    D is sqrt(DX * DX + DY * DY).

% поиск ближайшей задачи
найти_ближайшую_задачу(NewTrend, NewSeasonality, BestTask, BestDistance) :-
    findall(Distance-TaskID,
            (
            	задача(TaskID, Trend, Seasonality, _),
             	расстояние(NewTrend, NewSeasonality, Trend, Seasonality, Distance)
     		),
            Distances),
    keysort(Distances, Sorted),
    Sorted = [BestDistance-BestTask|_].


% рекомендация методов
рекомендация_методов(NewTrend, NewSeasonality, Methods) :-
    найти_ближайшую_задачу(NewTrend, NewSeasonality, BestTask, _),
    задача(BestTask, _, _, Methods).

