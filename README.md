# שכחה קטסטרופלית — שחזור בסביבת PyTorch

שחזור של:
> **"An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"**
> Goodfellow, Mirza, Xiao, Courville, Bengio — arXiv:1312.6211, 2015

---

## שאלת המחקר

> *האם המסקנות המרכזיות של גודפלו et al. (2015) — עליונות ה-Dropout על פני SGD במניעת שכחה קטסטרופלית, ותלות דירוג פונקציות האקטיבציה בסוג המשימה — ניתנות לשחזור בסביבת PyTorch מודרנית, תחת מגבלות חישוב של חומרה ביתית (8 ניסיונות במקום 25)?*

---

## ניווט מהיר

| מסמך | תוכן |
|---|---|
| [מבוא ושאלת המחקר](docs/introduction.md) | רקע, השערות, הגדרות מרכזיות, עבודות קשורות |
| [מתודולוגיה](docs/methodology.md) | מבנה הניסוי, חיפוש היפר-פרמטרים, סטיות מהמאמר |
| [תוצאות](docs/results.md) | כל 6 הגרפים עם ניתוח לכל תרחיש |
| [מסקנות](docs/conclusion.md) | טבלת השוואה כמותית, תשובה לשאלת המחקר, מגבלות |
| [תובנות](takeaways.md) | ניתוח לכל תרחיש ורפלקציה |

---

## מה משוחזר

המאמר מאמן רשתות MLP דו-שכבתיות על זוגות משימות ברצף ומודד את האיזון בין ביצועים על המשימה הישנה לעומת החדשה. שוחזרו שלושת התרחישים המקוריים:

| תרחיש | משימה ישנה | משימה חדשה | איורים במאמר |
|---|---|---|---|
| 1 — עיצוב קלט מחדש | MNIST | MNIST עם Permutation | Fig 1–2 |
| 2 — משימות דומות | ביקורות Amazon Kitchen | ביקורות Amazon DVD | Fig 3–4 |
| 3 — משימות שונות | MNIST (ספרות 2 ו-9) | ביקורות Amazon DVD | Fig 5–6 |

כל תרחיש כולל **8 תנאים** (4 פונקציות אקטיבציה × SGD / Dropout):
Sigmoid, ReLU, Maxout, LWTA — כל אחת עם ובלי Dropout.

---

## הגרפים המשוחזרים

הקו המקווקו האפור בכל גרף Frontier מסמן את חציון שגיאת המשימה הישנה בתחילת האימון על המשימה החדשה — נקודת הייחוס לפני השכחה.

| Fig 1 — Frontier, עיצוב קלט מחדש | Fig 2 — גודל מודלים |
|---|---|
| ![](paper_figures/Fig1_frontier_input_reformatting.png) | ![](paper_figures/Fig2_model_sizes_input_reformatting.png) |

| Fig 3 — Frontier, משימות דומות | Fig 4 — גודל מודלים |
|---|---|
| ![](paper_figures/Fig3_frontier_similar_tasks.png) | ![](paper_figures/Fig4_model_sizes_similar_tasks.png) |

| Fig 5 — Frontier, משימות שונות | Fig 6 — גודל מודלים |
|---|---|
| ![](paper_figures/Fig5_frontier_dissimilar_tasks.png) | ![](paper_figures/Fig6_model_sizes_dissimilar_tasks.png) |

---

## מבנה המאגר

```
final_experiment_repro.py   # ניסוי ראשי — כל 3 התרחישים
plot_results.py             # יצירת גרפי Frontier וגודל מודלים
prepare_amazon_npz.py       # עיבוד מקדים של נתוני Amazon → .npz
requirements.txt            # תלויות (Python 3.11)
takeaways.md                # ניתוח תרחישים ורפלקציה
docs/
  introduction.md           # שאלת מחקר, רקע, השערות, עבודות קשורות
  methodology.md            # עיצוב ניסויי וסטיות מהמאמר
  results.md                # תוצאות מפורטות עם כל 6 הגרפים
  conclusion.md             # טבלה כמותית, מסקנות, מגבלות
paper_figures/              # גרפים סופיים בשמות תואמי המאמר (Fig1–Fig6)
results_repro/              # checkpoints וגרפים מהריצה
```

---

## הוראות הרצה

**דרישות:** Python 3.11, CUDA 11.8+ (אופציונלי). זמן ריצה משוער: 6–12 שעות על GPU ביתי (RTX 3060 ומעלה), 24–48 שעות על CPU בלבד.

```bash
# 1 — התקנת תלויות
pip install -r requirements.txt

# 2 — הכנת נתוני Amazon (פעם אחת בלבד)
#     הורדה מ: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
#     מיקום: data/amazon/{books,dvd,electronics,kitchen}/
python prepare_amazon_npz.py

# 3 — הרצת הניסויים (שמירת checkpoint אוטומטית, המשך מנקודת עצירה)
python final_experiment_repro.py

# 4 — יצירת גרפים
python plot_results.py
```

נתוני MNIST מורדים אוטומטית בהרצה הראשונה.

---

## ממצאים מרכזיים

- **Dropout מציג עקומות Frontier עדיפות** על SGD בכל 3 התרחישים — best_joint נמוך יותר ב-6 מתוך 8 תנאים בתרחיש 1.
- **Maxout + Dropout** הוא התנאי היחיד המופיע על ה-Frontier בכל שלושת התרחישים — עקבי עם טענת המאמר המקורי.
- **דירוג פונקציות האקטיבציה תלוי-תרחיש** — אין מנצח אוניברסלי; cross-validation הכרחי.
- לפרטים כמותיים מלאים ראו [מסקנות](docs/conclusion.md).
