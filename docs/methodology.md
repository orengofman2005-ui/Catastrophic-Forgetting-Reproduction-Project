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

## חיפוש היפר-פרמטרים

חיפוש אקראי (random search) — 8 ניסיונות לכל condition (המאמר: 25):

| פרמטר | טווח חיפוש |
|---|---|
| Learning rate | `10^U[-2.0, -0.5]` |
| Hidden layer size | `U[250, 5000]` |
| Max-norm constraint | לפי נספחי המאמר |
| Momentum | עולה ליניארית מ-0.5 |

## סטיות מהמאמר המקורי

| פרמטר | מאמר | כאן | סיבה |
|---|---|---|---|
| Trials per condition | 25 | 8 | חומרה ביתית |
| Early-stopping patience | 100 | 15 | חומרה ביתית |
| Framework | Theano / Pylearn2 | PyTorch | Theano deprecated |
| Batch size | 128 | 256 | GPU utilization |

## Checkpoint ו-Resume

הסקריפט שומר checkpoint אחרי כל condition. אם הריצה נקטעת — היא ממשיכה אוטומטית מהמקום האחרון שנשמר.
