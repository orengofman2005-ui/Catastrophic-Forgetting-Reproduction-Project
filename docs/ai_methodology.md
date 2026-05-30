# AI Usage & Methodology

## Overview

This document describes how AI-assisted tools — primarily Claude (Anthropic) — were used during the project.
**הערה:** זה הקורס הראשון שלי בתחום. השימוש ב-Claude היה נרחב מאוד לאורך כל הפרויקט — שאלות על הבנת המאמר, פרשנות הגרפים, החלטות מתודולוגיות, והבנת קונספטים (Maxout, LWTA, Pareto frontier, LR decay וכו׳). ראו את כל השאלות בפירוט ב-[`ai_plans.md`](ai_plans.md).

## Tools Used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Python basics, full code generation, debugging, architecture design, paper interpretation |
| GitHub Copilot | Minor autocompletion |

## Workflow

```
קריאת המאמר → שאלות בסיסיות על Python → Claude מייצר קוד → בדיקה ותיקון עם Claude
       ↑                                                              ↓
  חזרה אחורה ←  השוואה לנתוני המאמר ← הרצת ניסוי ← ניפוי שגיאות עם Claude
```

1. **קריאת המאמר קודם** — המאמר המקורי של Goodfellow et al. (2013) נקרא לפני כתיבת כל קוד. הגדרות עיקריות (dropout rates, early-stopping patience, frontier computation) חולצו ידנית.

2. **שאלות על הפרויקט** — מכיוון שזה הקורס הראשון שלי בתחום, חלק ניכר מהשיחות עם Claude עסק בהבנת הקונספטים: מה זה Pareto frontier, מה ההבדל בין Maxout ל-LWTA, למה משתמשים ב-joint early stopping, מה עושים כשגרף לא נראה נכון. ראו את כל השאלות בקובץ [`ai_plans.md`](ai_plans.md).

3. **Claude מייצר קבצי קוד מלאים** — לאחר שהבנתי מה צריך לבנות, Claude יצר קבצי Python שלמים (ראו "What WAS AI-Generated" למטה). הקבצים נקראו שורה-שורה ואומתו מול המאמר.

4. **תיקון איטרטיבי** — כל פעם שגרף לא נראה נכון, או שקיבלתי שגיאה שלא הבנתי, חזרתי ל-Claude עם הבעיה הספציפית.

## What WAS AI-Generated

הקבצים הבאים נוצרו על ידי Claude ושולבו ישירות בפרויקט:

### `pytorch_reproduction_suite.py`
קובץ Python מלא שנוצר על ידי Claude. כולל:
- Data loaders לשלושת התרחישים (Permuted MNIST, Amazon Reviews, MNIST+Amazon)
- `CatastrophicForgettingMLP` עם ReLU / Sigmoid / Maxout / LWTA
- Hyperparameter sampling התואם את טווחי החיפוש של המאמר
- Per-layer max column-norm constraint
- לולאת אימון עם 25 trials × 8 conditions
- חישוב Pareto lower-left frontier וגרף log-log

### `final_experiment_repro.py`
קובץ Python מלא שנוצר על ידי Claude כגרסה משופרת:
- Checkpoint נשמר אחרי כל condition (המשך אוטומטי במקרה של קריסה)
- `tqdm` progress bars בזמן אמת
- `TruncatedSVD` לצמצום מימד עבור תרחיש 3
- טווחי hyperparameter מדויקים מנספחי המאמר

### `plot_results.py`
קובץ Python מלא שנוצר על ידי Claude להפקת גרפים בסגנון המאמר.

## What Was NOT AI-Generated

- ההחלטה אילו 3 תרחישים להריץ ואיזה 4 פונקציות אקטיבציה להשוות.
- הוולידציה שהמגמות בגרפים שלנו תואמות לאיכות הגרפים במאמר.
- הניתוח הכתוב ב-[`takeaways.md`](../takeaways.md).
- ההחלטה להשתמש ב-TruncatedSVD (ולא padding) לצמצום מימד בתרחיש 3.
