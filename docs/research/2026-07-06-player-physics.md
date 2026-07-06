# Физика игрока: что уже стоит, что жалобы вскрывают, что делать (2026-07-06)

Готово для совместного обсуждения — не имплементация, а карта уровня «сначала договоримся, потом кодим».

**Триггер:** пользовательский отчёт 2026-07-06 — «игроки висят в воздухе, двигаются неестественно, очень быстро меняют позы, неестественно быстро перемещаются по полю, ориентация в пространстве, появляются из ниоткуда и пропадают, пролетают сквозь предметы, нет предела выносливости».

**Формат:** для каждого симптома — измеримое определение, что уже покрыто в коде, что осталось незакрытым, вариант фикса с оценкой стоимости и приоритетом.

---

## 1. Разложение жалоб пользователя на измеримые пределы

Сгруппировано по типу нарушения, чтобы не путать «этой уже есть» с «ещё нет».

| # | Жалоба | Что это в терминах модели | Измерение | Реальный предел (elite футбол) |
|---|--------|--------------------------|-----------|-------------------------------|
| A | Висят в воздухе | root Z над пробитой землёй Z=0 | `transl[:, 2]` vs `pelvis_height_m ≈ 0.92 м` | Z должен быть **колеблющимся 0.85–1.05 м** во время бега; для прыжка — эпизод <0.5 с, пик +0.3 м; никогда постоянно >1.10 м |
| B | Неестественно быстро перемещаются | root XY speed | `|d(transl[:2])/dt|` | Bolt 12 м/с спринт; elite футболист **пик 10 м/с (36 км/ч)**, устойчиво <8 м/с |
| C | Быстро меняют позы | joint angular velocity | `|Δ(body_pose)/dt|` per joint | Плечо ~1500°/с (баскет-бросок), большинство суставов **<600°/с в футболе** |
| D | Ориентация в пространстве прыгает | root global_orient rate | `|Δ(global_orient)/dt|` | Elite поворот тела до **720°/с** только эпизодически; sustained turn ~360°/с |
| E | Появляются/пропадают | subject presence на границах и в дырах | `frame_conf` и края span | Игрок физически на поле от свистка до свистка — исчезнуть может только за пределами кадра |
| F | Пролетают сквозь предметы | collision player↔player, player↔ball | попарные расстояния capsule | Тела не пересекаются (реалистичный минимум ~0.5 м между центрами) |
| G | Нет предела выносливости | долговременная скорость | доля времени с speed>threshold | Elite игрок держит спринт **<10 с**, между спринтами восстанавливается ~30–60 с; общий high-intensity <3–5% времени |

Все семь пунктов — это разные физические измерения. Каждый требует своего gate/correction; попытка «один умный сглаживатель для всего» уже была (MA(5)) и структурно не срабатывает для teleport-класса ошибок (см. §2).

---

## 2. Что УЖЕ стоит в коде (не переоткрывать)

### 2.1 M3-9 kinematic gate — `src/pitch3d/core/correction/kinematics.py`

**Что делает** (уровень поверхности, чтобы не читать 331 строку с нуля):

* Порог **max_speed 10.5 м/с** и **max_accel 8.0 м/с²** (константы `HUMAN_MAX_SPEED`, `HUMAN_MAX_ACCEL`; env-override `PITCH3D_KIN_MAX_SPEED`/`MAX_ACCEL`).
* Проекция XY-трека на feasible-set: velocity clamp → чередующиеся forward/backward accel sweeps → финальный forward sweep с гарантией допустимости. Оба конца сегмента прикреплены к измеренным позициям.
* **Teleports MARKED, not erased** (R-6): единичный интервал скорости >2× лимита → `TeleportEvent`, jump сохраняется как есть, помечается для ID-review. Одноразовые out-and-back spikes демотятся до jitter (тест на разворот вектора).
* Consecutive teleport-интервалы схлопываются в ОДНУ область.
* Результат — одна плотная `KEYFRAME_INTERP` correction на игрока через ADR-0002 seam (inspectable, disableable, non-destructive).

**Реальный результат** (см. STATUS.md строка 252): скоростные/accel violations 22/999 → **0/0**, 10 raw teleports → 1 marked region (subj 1 f31 8.7 м, n_intervals=8, conf 0.2 = coherence-extrapolated).

**Область покрытия:** только **root XY**. Z, pose, orientation — не трогает.

### 2.2 Coherence coast/gap-fill — `src/pitch3d/core/correction/coherence.py`

* **Gap-fill (`fill_pose_gaps`)**: интерьерные дыры ≤ `max_fill_gap=12` кадров зашиваются slerp'ом (rotation) + linear (translation), помечаются `filled_confidence=0.3`.
* **Edge extension (`extend_pose_to_span`)**: игроки, потерянные раньше или позже клипа, продлеваются до полного span'а. Поза заморожена, root коастит с decaying velocity, `extrapolated_confidence=0.2`.
* **Coast velocity capped** на `coast_max_speed=HUMAN_MAX_SPEED=10.5 м/с` (fix #207: без cap'а умирающий трек передавал 43 м/с наследство → 10.9 м skid).
* **Auto smoothing correction** (MA(5) или gaussian): на root_translation по умолчанию, на root_orientation — off (может задавить резкие развороты).

### 2.3 Foot-ground anchor — `src/pitch3d/adapters/models/pose.py`

* `_ground_root` (l.256): XY берётся из homography (bbox-bottom → world), Z = либо `pelvis_above_foot` из бекенда (varying), либо константа `pelvis_height_m ≈ 0.92 м`.
* `refit(constraints=…)` умеет `foot_floor` (clamp root Z ≥ floor) — но применяется только когда явно передано (в текущем экспорте — не автомат).

### 2.4 Ball physics — DONE (#206)

Контактный anchoring: там где мяч касается ноги, XY пришиты к футу, между — ballistic Z. Мяч чистый (measured p95=16.2 м/с, 0 нарушений). Не трогаем.

### 2.5 Probe — `scripts/motion_stats.py`

Пишет per-subject speed/accel/turn на raw proposal И resolved (proposal ⊕ corrections). Запускать: `python scripts/motion_stats.py --scene out/anim_adr11/export/scene.json --fps 29.97`.

---

## 3. Разбор жалоб — что НЕ покрыто

Для каждого пункта из §1: **что фактически происходит** + **почему текущий stack это пропускает**.

### A) «Висят в воздухе»

**Скорее всего происходит:** `pelvis_above_foot` из HMR-бекенда стабильный ≈ 0.92 м, gate накладывает feasible XY, но НЕ проверяет что foot действительно касается плоскости Z=0. Если homography слегка неверна (метровый offset в U-V→world) или SMPL-X shape крупнее нормы, ноги висят на 10–30 см над землёй **постоянно** — глазом читается как «вертолёт».

**Почему пропускается:**
- `_ground_root` использует `bbox bottom = foot`, но HMR может вернуть pose где foot ≠ bbox-bottom (ботинки внутри маски).
- `foot_floor` constraint существует в `refit()`, но в текущей `pipeline.py` для рендера не выставляется автоматически.
- Никто не измеряет «доля кадров с foot z > ε» в motion_stats.

**Как проверить прямо сейчас:** прогнать на сцене `python -c "import numpy as np; from pitch3d.core.scene.serialization import load_scene; s = load_scene(...); [print(sub.track_id, sub.proposal.pose.transl[:,2].min(), sub.proposal.pose.transl[:,2].max()) for sub in s.subjects]"` — если min/max Z вокруг 0.92 без движения → foot не приземлён.

### B) «Неестественно быстро перемещаются»

**Скорее всего происходит:** M3-9 gate работает **на root XY**, но:
- Порог `HUMAN_MAX_SPEED=10.5` — это ПИК спринтера, а не крейсер. Устойчивый бег 6–8 м/с. Возможно gate пропускает 8–10 м/с длинные ID-swap slides (не ловится spike-test'ом).
- Если tracker передаёт «дрейф камерой» вместо реального движения, gate его сглаживает, но глаз читает как «слишком быстро всё сдвинулось».
- **FPS mismatch:** mux `FPS=25` vs source `29.97` (STATUS.md 252) — плейбек на 20% медленнее реалтайма. То есть если мы КОРРЕКТНО считаем при 29.97, а показываем при 25, — глаз видит **замедленное** движение. Если жалоба «слишком быстро», проблема не там, но fps consistency стоит проверить.

**Как проверить:** `scripts/motion_stats.py --fps 29.97` до/после gate + сверить с mux fps.

### C) «Быстро меняют позы»

**Скорее всего происходит:** joint angular velocity **не ограничен**. HMR даёт per-joint axis-angle на каждый кадр; при tracker jitter конечности дёргаются между кадрами (типичный neural HMR failure mode при частичной occlusion).

**Почему пропускается:** ни `kinematics.py`, ни `coherence.py` не смотрят на body_pose. Coherence smoothing на root_translation **не касается joints**. Slerp — только для gap-fill интерьера.

**Что нужно:** отдельный gate на per-joint angular velocity (или per-joint acceleration в quaternion domain). Первая версия — простой **per-joint slerp с ограничением по quaternion distance между кадрами** (аналог max_speed для root, но для rotation). Порог типа 600°/с × dt = 20° на 30 fps.

### D) «Ориентация в пространстве»

**Скорее всего происходит:** root global_orient (rotation тела в мире) может флипаться на 180° между кадрами, если HMR неопределён по фронт/спине. Или resonant jitter при близких камере-игроке позах.

**Почему пропускается:** `CoherenceConfig.smooth_root_orientation=False` (по дефолту off, комментарий: «can over-flatten fast turns»).

**Что нужно:** аналог M3-9 но для global_orient — **max_turn_rate** гейт с раздельными порогами (720°/с для fast turns, marked как spike выше). И включить `smooth_root_orientation=True` с малым окном (3–5 кадров) как auto default; исключать marked spikes из smoothing'а.

### E) «Появляются из ниоткуда и пропадают»

**Уже частично покрыто** (`extend_pose_to_span`): игрок продлевается на весь span клипа, holds posture + coasts root. Но:
- **Может рендер их не отрисовывать** если `subject_frame_conf < threshold` — надо проверить, не блинкает ли что-то в `blender_animate.py` по confidence.
- **Появление в середине** (новый ID) всё равно даст «из ниоткуда» если не смёржен с pre-track continuity.
- **Заявленное сегодня в §6:** 1 marked teleport region (subj 1 f31, 8.7 м) — это как раз «пропал и появился». Marked, но НЕ ИСПРАВЛЕН — R-6 (мы не хотим fabricate тропу спринта).

**Что нужно:** проверка «рендерятся ли extrapolated frames», плюс политика для marked teleport regions (мигание vs plausible interpolation с explicit R-6 tag).

### F) «Пролетают сквозь предметы»

**НЕ покрыто вообще.** Игроки — независимые SMPL-X меши без collision. Мяч — тоже независимая точка (без mesh-collision, только контактный anchoring).

**Что нужно:** дешёвая capsule-collision (одна цилиндр-капсула ~0.5 м радиус × pelvis_height на игрока), soft-repulsion между пересечёнными парами. Не rigid body simulation — просто «pull them apart» iteration после kinematic gate. Ball collision — сложнее (есть моменты когда игрок должен ЛОГИЧЕСКИ пнуть мяч, — это уже M3-2 territory).

### G) «Нет предела выносливости»

**НЕ покрыто.** Никакого fatigue-модели нет.

**Что нужно (низкий приоритет):** running average скорости на скользящем окне (30 с?) — если >N% времени в high-intensity zone, следующий кадр deprioritize sprint-класс движения. НО: это скорее «prior для motion smoothing» чем hard gate — реальные игроки перегружены нередко.

**Мой рекомендация:** отложить. Из семи жалоб это самая эстетическая; она не даст eye-visible improvement пока не закрыты A–D. Если и делать, то как feature-vector, не как hard clamp.

---

## 4. Предлагаемый порядок работы (что дать пользователю сейчас vs потом)

Взвешен по «глазу» (влияет ли на итоговый видос), измеримости и стоимости имплементации.

### Tier 0 — измерить прежде, чем чинить

**T0.** Прогнать `scripts/motion_stats.py` на текущей сцене + расширить его: добавить `foot_z_stats` (min/max/фракция z>ε), `joint_omega_stats` (per-joint angular velocity), `turn_rate_stats` (global_orient rate).

*Оценка: 1 итерация. Стоимость: локально, $0.*

**Выход:** таблица «фактических нарушений по 7 категориям» на реальной сцене. Без этого мы будем чинить по ощущениям.

### Tier 1 — покрыть очевидные незакрытые (A, C, D)

**T1.a — Foot floor auto-default.** Включить `foot_floor=0.0` в default constraints для рендер-пайплайна, не только по явному запросу. Плюс sanity gate «foot_z > 0.30 → warn» (что-то не так с shape/homography).

*Оценка: 1 итерация. Стоимость: локально, $0. Пробуем на существующей сцене.*

**T1.b — Per-joint angular gate.** Новый модуль `core/correction/joint_kinematics.py`: max_omega_deg_per_s (600°/с per joint как первая версия), quaternion-slerp clamp, marked как spikes с R-6. Один `KEYFRAME_INTERP` per joint per subject через ADR-0002 seam, тесты по паттерну M3-9.

*Оценка: 2–3 итерации. Стоимость: локально, $0 unit-тесты; pod E2E $0.10 для eye-verify.*

**T1.c — Root orientation gate.** Расширить `KinematicConfig` с `max_turn_rate_deg_per_s=720`; включить `smooth_root_orientation=True` c исключением marked spikes.

*Оценка: 1–2 итерации. Стоимость: $0.*

### Tier 2 — presence / persistence hardening (E)

**T2.a — Rendering audit.** Убедиться что `extrapolated` кадры реально рендерятся (не блинкают). Проверить `blender_animate.py` на condition по `subject_frame_conf`.

**T2.b — Marked teleport interpolation policy.** Опция: `teleport_policy=hold|interpolate|flash` — по умолчанию `hold` (без изобретения motion), с флагом для плавного `interpolate` (marked R-6).

*Оценка: T2.a — 1 итерация; T2.b — 1–2 итерации. $0.*

### Tier 3 — collision (F)

**T3.a — Capsule collision.** Пост-процесс над resolved motion: пересечения капсул → soft push apart. Не физическая симуляция — Jacobi iteration.

*Оценка: 3–5 итераций (нужны хорошие defaults). $0 unit, $0.10 pod.*

### Tier 4 — fatigue (G)

Откладываем. Сначала A–F, затем оцениваем есть ли нужда.

---

## 5. Открытые вопросы для обсуждения

1. **Foot-floor политика:** hard clamp к Z=0 (жёстко «не проваливать»), soft attractor (blend), или адаптивный (foot_z<ε → free, foot_z>ε → clamp)? Прыжки хочется сохранить.
2. **Per-joint gate — на raw HMR или на post-coherence?** Coherence уже сглажил через MA(5); двойное сглаживание может задавить настоящие быстрые движения (удар по мячу).
3. **Teleport policy:** после жалобы «появляются из ниоткуда» — стоит ли по умолчанию switch'нуться с `hold` на `interpolate` (marked R-6)? Или это стёрет legitimate ID-swap для стич-ревью?
4. **FPS mismatch:** зафиксировать mux fps на source (29.97 в стриме) как основной фикс восприятия скорости, или сохранять 25 как удобное совместимое число?
5. **Collision между игроками vs между игроком и мячом:** ball-touch УЖЕ используется в `ball_lift.py` для anchor. Стоит ли расширять до полной collision, или ограничиться player↔player?
6. **Ceiling:** 10.5 м/с — elite ceiling. Для матча Colombia–DR Congo — не олимпийцы. Опустить до 9 м/с default? (env-override уже есть.)

---

## 6. Ссылки на существующий код

- `src/pitch3d/core/correction/kinematics.py` — M3-9 gate (XY only).
- `src/pitch3d/core/correction/coherence.py` — gap-fill + edge extend + coast cap + auto smoothing.
- `src/pitch3d/core/correction/engine.py` — resolve_subject_motion (композиция proposal ⊕ corrections).
- `src/pitch3d/adapters/models/pose.py::_ground_root` (l.256), `refit(foot_floor=…)` (l.231).
- `src/pitch3d/core/orchestration/ball_lift.py` — contact-anchored ball (#206 закрыт).
- `scripts/motion_stats.py` — probe.
- `docs/STATUS.md` строка 252 (#207 таблица) — самое подробное описание истории вопроса.
- `docs/roadmap.md` строка 500 — карточка M3-9 в roadmap.

---

## 7. Мини-план (если вы согласны с приоритетами)

1. **Сейчас:** расширить `motion_stats.py` (T0), прогнать на текущей сцене, вернуться с таблицей фактов.
2. **Далее:** T1.a foot floor auto-default (быстрый визуальный выигрыш).
3. **Далее:** T1.b joint angular gate (закрывает «быстро меняют позы»).
4. **Далее:** T1.c orientation gate (закрывает «ориентация в пространстве»).
5. **Далее:** T2 presence audit + policy.
6. **Дальше по обстановке:** collision, fatigue.

Готов приступить к T0, если вы согласны, или перекроить приоритеты.
