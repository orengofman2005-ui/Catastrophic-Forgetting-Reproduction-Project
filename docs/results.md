# תוצאות השחזור

## גרפי Possibilities Frontier

כל גרף מציג את עקומת הגבול התחתון (lower convex hull) של כל 8 השיטות.
ציר X = שגיאה על המשימה הישנה, ציר Y = שגיאה על המשימה החדשה (שניהם לוגריתמי).
**נקודות קרובות לראשית הצירים = ביצועים טובים בשני הממדים.**

---

### תרחיש 1 — Input Reformatting (MNIST → Permuted MNIST)

![Frontier](../paper_figures/Fig1_frontier_input_reformatting.png)
![Model Sizes](../paper_figures/Fig2_model_sizes_input_reformatting.png)

שני הגרפים תואמים את Figure 1 ו-Figure 2 במאמר.
המשימות זהות מבחינה מבנית אך עם permutation שונה של הפיקסלים.

> **הערה לקורא:** איור 2 מדגים כי תחת תנאי Dropout, הארכיטקטורות שנבחרו כביצועיות ביותר הן בעלות קיבולת פרמטרים מורחבת — בפרט Maxout ו-LWTA. כפי שנראה באיור 1, קיבולת זו מתורגמת ישירות לחסינות משופרת מפני שכחה (עקומות Frontier יציבות וקרובות יותר לראשית), מה שמאשש את טענת המאמר לפיה Dropout מאפשר אימון רשתות רחבות החסינות יותר לשכחה קטסטרופלית.

---

### תרחיש 2 — Similar Tasks (Amazon Kitchen → Amazon DVD)

![Frontier](../paper_figures/Fig3_frontier_similar_tasks.png)
![Model Sizes](../paper_figures/Fig4_model_sizes_similar_tasks.png)

שני הגרפים תואמים את Figure 3 ו-Figure 4 במאמר.
שתי משימות ניתוח סנטימנט על קטגוריות שונות של Amazon.

---

### תרחיש 3 — Dissimilar Tasks (MNIST → Amazon DVD)

![Frontier](../paper_figures/Fig5_frontier_dissimilar_tasks.png)
![Model Sizes](../paper_figures/Fig6_model_sizes_dissimilar_tasks.png)

שני הגרפים תואמים את Figure 5 ו-Figure 6 במאמר.
משימות שונות לגמרי — ראייה ממוחשבת vs. עיבוד שפה טבעית.

> **הערה לקורא:** שימו לב לאנומליה באיור 6 — מודל ה-LWTA מוקצה עם כ-9 מיליון פרמטרים. עם זאת, איור 5 מוכיח כי למרות הקיבולת העצומה, המודל חווה קריסה מהירה ושכחה קטסטרופלית חריפה במעבר למשימה הלא-דומה. ממצא זה מבהיר לקורא כי הגדלת קיבולת המודל לבדה אינה מנגנון הגנה מספק כאשר המשימות שונות לחלוטין מבחינה סמנטית.

---

## ממצאים מרכזיים

- **Dropout** — מככב בכל 3 התרחישים. נותן את האיזון הטוב ביותר בין שמירה על המשימה הישנה לבין ביצועים במשימה החדשה.
- **Maxout + Dropout** — השיטה היחידה שמופיעה על ה-frontier בכל שלושת התרחישים.
- **פונקציית האקטיבציה** — דירוגה תלוי בתרחיש; אין "מנצח" אוניברסלי.
- **LWTA** — לא מכה את ה-SGD baseline באופן עקבי (בניגוד לטענות Srivastava et al. 2013).
