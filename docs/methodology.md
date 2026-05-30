# מתודולוגיה ניסויית

## מבנה הניסוי

הניסוי עוקב אחר המבנה המקורי של גודפלו et al. (2015):

1. **אימון על משימה ישנה** — עד להתכנסות (early stopping על validation set)
2. **אימון על משימה חדשה** — תוך מדידת שני מדדים במקביל:
   - שגיאה על המשימה החדשה (ציר Y)
   - שגיאה על המשימה הישנה (ציר X)
3. **ציור עקומת Possibilities Frontier** — הגבול התחתון של ענן הנקודות (lower convex hull), בסקאלה לוגריתמית

## 8 תנאים (Conditions)

| פונקציית אקטיבציה | אלגוריתם |
|---|---|
| Sigmoid | SGD |
| Sigmoid | Dropout |
| ReLU | SGD |
| ReLU | Dropout |
| Maxout | SGD |
| Maxout | Dropout |
| LWTA | SGD |
| LWTA | Dropout |

## ארכיטקטורת המודל

- **שכבות:** 2 שכבות hidden + softmax classification layer
- **Maxout:** pool size $k=2$ (כל יחידת פלט היא מקסימום על 2 קלטים)
- **LWTA:** group size $k=2$ (בכל זוג יחידות, רק הגדולה מקבלת גרדיאנט)
- **Max-norm constraint:** מגבלה דינמית לכל שכבה בטווח 1.0–5.0, נדגמת בנפרד עבור `fc1`, `fc2`, `fc_out` בכל ניסוי
- **Dropout:** הסתברות dropout_hidden=0.5, dropout_input=0.2 (קבועים, לא חלק מחיפוש)

## חיפוש היפר-פרמטרים

חיפוש אקראי (random search) — 8 ניסיונות לכל condition (המאמר: 25):

| פרמטר | טווח חיפוש |
|---|---|
| Learning rate | `10^U[-2.0, -0.5]` |
| Hidden layer size | `U[250, 5000]` |
| Max-norm (כל שכבה) | `U[1.0, 5.0]` |
| Weight init range (irange) | `10^U[-2.3, -1.0]` |
| Sparse init k (Sigmoid/ReLU) | `U[10, 30]` |
| Momentum | עולה ליניארית מ-0.5 |

> Maxout ו-LWTA: bias מאותחל ל-0 (אתחול רנדומי של bias גורם לדומיננטיות של יחידה אחת בקבוצה).
> Sigmoid: bias מאותחל מטווח שלילי לעידוד sparsity.
> ReLU: bias חיובי קל למניעת יחידות "מתות".

## סטיות מהמאמר המקורי

| פרמטר | מאמר | כאן | סיבה |
|---|---|---|---|
| Trials per condition | 25 | 8 | חומרה ביתית |
| Early-stopping patience | 100 epochs | 15 epochs | חומרה ביתית |
| Framework | Theano / Pylearn2 | PyTorch | Theano deprecated |
| Batch size | 128 | 256 | GPU utilization |

## Checkpoint ו-Resume

הסקריפט שומר checkpoint אחרי כל condition. אם הריצה נקטעת — היא ממשיכה אוטומטית מהמקום האחרון שנשמר.
