# חשיבה אלגוריתמית – Catastrophic Forgetting Reproduction Project

מסמך זה מפרט את הלוגיקה המודולרית של הפרויקט, שלב אחר שלב, כולל פרוטוקולי האימות שנעשו בכל שלב.

---

## שלב 1 — עיבוד מקדים של הנתונים (`prepare_amazon_npz.py`)

### מה עושה המודול
טוען את מערך Amazon Reviews (4 קטגוריות: Kitchen, DVD, Electronics, Books), מבצע Bag-of-Words vectorization, ומייצא קובץ `.npz` לכל קטגוריה.

### לוגיקת המימוש
1. טעינת קבצי ה-review הגולמיים מפורמט XML.
2. חילוץ תוית החיוביות/שליליות מכל ביקורת.
3. בניית וקטור TF או Binary BoW מהטקסט.
4. שמירה בפורמט NumPy (`X_train`, `y_train`, `X_test`, `y_test`).

### פרוטוקול אימות
- בדיקה שמימדי המטריצות תקינים: `X_train.shape[1]` זהה בין הקטגוריות.
- וידוא שהתוויות בינאריות (0/1 בלבד).
- בדיקה שאין ערכי NaN או Inf בוקטורים.

---

## שלב 2 — ארכיטקטורת הרשת (`final_experiment_repro.py` — Sections 2–3)

### מה עושה המודול
מגדיר את מחלקת `MLP` (Multi-Layer Perceptron) עם שתי שכבות נסתרות, ותומך בארבע פונקציות אקטיבציה: ReLU, Sigmoid, Maxout, LWTA.

### לוגיקת המימוש

**פונקציות אקטיבציה:**
- **ReLU / Sigmoid** — שכבות סטנדרטיות של PyTorch.
- **Maxout** — כל יחידה מחשבת מקסימום על קבוצת `pool_size=2` נוירונים. מיושם בידי פיצול המימד ולקיחת max לאורך ציר ה-pool.
- **LWTA (Local Winner Takes All)** — דומה ל-Maxout, אך משתמש ב-mask בינארי: רק הנוירון המנצח שומר את ערכו, האחרים מתאפסים. נוסף רעש אקראי קטן (`1e-6`) לשבירת תיקו.

**Dropout לפי המאמר (§3.1):**
- שכבת הקלט: `p=0.2`
- שכבות נסתרות: `p=0.5`
- כאשר `use_dropout=False` — נעשה שימוש ב-`nn.Identity()` במקום.

**אתחול משקלים:**
- Maxout / LWTA: `Uniform(-0.005, 0.005)` — כפי שנצפה בספרות.
- Sigmoid: `Xavier Uniform` לשכבות, `Uniform(-0.2, 0)` לביאסים.
- ReLU: `Kaiming Uniform` או `Xavier Uniform` לפי דגימת ה-hyperparameter.

### פרוטוקול אימות
- בדיקה שמספר הפרמטרים הכולל סביר לפי ה-hidden_dim שנבחר.
- ריצת forward pass על batch קטן וווידוא שה-output shape תקין.
- בדיקה שב-LWTA בדיוק נוירון אחד מנצח לכל קבוצה (סכום ה-mask = 1).

---

## שלב 3 — לולאת האימון ו-Early Stopping (Section 5)

### מה עושה המודול
מאמן את הרשת על משימה ראשונה עד להתכנסות, ואז מאמן על משימה שנייה תוך מעקב אחרי הביצועים על שתי המשימות.

### לוגיקת המימוש

**אימון משימה 1 (`train_task1`):**
```
for epoch in range(MAX_EPOCHS_OLD):
    train one epoch
    val_err = evaluate on validation set
    if val_err improved: save best_state, reset patience counter
    else: increment patience counter
    if patience counter >= PATIENCE_OLD (100): stop
restore best_state
```

**אימון משימה 2 (`train_task2_and_log`):**
```
for epoch in range(MAX_EPOCHS_NEW):
    train one epoch on task 2
    old_val = error on task 1 validation
    new_val = error on task 2 validation
    joint = old_val + new_val  # שמירת האיזון בין שתי המשימות
    if joint improved: save best_state, reset patience counter
    log (old_test_error, new_test_error) for Pareto frontier
    if patience counter >= PATIENCE_NEW (100): stop
restore best_state
```

**Max-Norm Constraint:**
אחרי כל optimizer step, מוחל הconstraint על כל שכבת `nn.Linear`:
```python
norms = weight.norm(2, dim=1, keepdim=True)
weight *= clamp(norms, max=max_norm) / (norms + epsilon)
```

### פרוטוקול אימות
- וידוא שה-val error יורד בשלב האימון הראשון (אינדיקציה לכך שהמודל לומד).
- בדיקה שמנגנון ה-checkpoint שומר ומשחזר נכון את `best_state`.
- ריצת תרחיש קצר (5 trials, patience=10) לבדיקת סיום תקין לפני הרצה מלאה.

---

## שלב 4 — חיפוש Hyperparameters (Section 6)

### מה עושה המודול
לכל combination של activation (4) × dropout (2) = 8 תנאים, מריץ 25 trials עם hyperparameters שנדגמים באקראי, ושומר checkpoint אחרי כל תנאי.

### לוגיקת המימוש

**מרחב החיפוש:**
| פרמטר | טווח |
|--------|------|
| hidden_dim | [128, 256, 512, 800, 1024, 1200, 1600, 2048] |
| learning_rate | LogUniform(1e-4, 1e-1) |
| momentum | Uniform(0.3, 0.99) |
| max_norm | {1.0, 1.5, 2.0, 3.0, 4.0, 5.0} |
| init_name | {xavier, kaiming, uniform} |

**מנגנון Checkpoint:**
- לפני הרצת כל תנאי, נבדק אם קיים קובץ `ckpt_{scenario}_{label}.pt`.
- אם קיים — התנאי מדולג ותוצאותיו נטענות.
- אם לא — התנאי מורץ ותוצאותיו נשמרות בסיומו.

### פרוטוקול אימות
- וידוא שה-seed קבוע לכל trial (`SEED + trial_number`) לצורך reproducibility.
- בדיקה שמספר ה-trials שהסתיים שווה ל-25 לכל תנאי.
- וידוא שהcheckpoints נטענים נכון ושהתוצאות זהות בין הרצות.

---

## שלב 5 — חישוב Pareto Frontier וויזואליזציה (Section 7)

### מה עושה המודול
מחשב את מעטפת ה-Pareto (tradeoff) בין שגיאת המשימה הישנה לשגיאת המשימה החדשה, ומציג אותה בגרף לוגריתמי.

### לוגיקת המימוש

**`pareto_lower_left`:**
1. קבלת כל נקודות ה-(old_error, new_error) מכל ה-trials.
2. סינון נקודות לא חוקיות (NaN, Inf, שגיאות אפסיות).
3. המרה לסקאלה לוגריתמית.
4. מיון לפי old_error.
5. בניית המעטפת: נכנסת נקודה רק אם new_error שלה קטן מכל הנקודות שלפניה.

**גרף:**
- ציר X: שגיאת מבחן על המשימה הישנה (סקאלה לוגריתמית).
- ציר Y: שגיאת מבחן על המשימה החדשה (סקאלה לוגריתמית).
- כל תנאי (activation × dropout) מוצג בצבע ומרקר ייחודיים.

### פרוטוקול אימות
- בדיקה שהמעטפת מונוטונית (ככל שה-old_error גדל, ה-new_error קטן).
- השוואה ויזואלית של הגרפים המופקים לאיורים 1, 3, 5 במאמר המקורי.
- וידוא שכל 8 התנאים מופיעים בגרף (אם אחד חסר — סימן לבעיה בנתונים).

---

## סיכום זרימת הפרויקט

```
prepare_amazon_npz.py
        ↓
  [kitchen.npz, dvd.npz, ...]
        ↓
final_experiment_repro.py
  ├── Scenario 1: Permuted MNIST → Permuted MNIST
  ├── Scenario 2: Amazon Kitchen → Amazon DVD
  └── Scenario 3: MNIST (2/9) → Amazon DVD
        ↓
  [results_fixed/*.csv, checkpoints]
        ↓
plot_results.py
        ↓
  [figure1.png, figure2.png, figure3.png]
```

כל שלב בזרימה תלוי בתוצאות השלב הקודם, ולכן הסדר חשוב להרצה תקינה.
