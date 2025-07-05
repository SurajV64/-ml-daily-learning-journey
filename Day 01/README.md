# 📘 Day 1: Underfitting vs Overfitting vs Well-Fitting — #40DaysOfML

Welcome to Day 1 of my **#40DaysOfML** challenge!  
Today, I explored the core concepts of **model fitting in machine learning**, which are crucial for building reliable predictive models.

---

## ✅ Topics Covered

- 🔹 Underfitting
- 🔹 Overfitting
- 🔹 Well-Fitting
- 🔹 Real-world analogies
- 🔹 Python implementation using scikit-learn
- 🔹 Visual analysis using `matplotlib`

---

## 📌 Real-World Analogy

| Scenario        | Example                                                   |
|-----------------|-----------------------------------------------------------|
| Underfitting     | Student reads only titles and fails in exams             |
| Overfitting      | Student memorizes answers, fails if questions change     |
| Well-Fitting     | Student understands concepts and answers any variation   |

---

## 💡 Concept Summary

| Model Type    | Train Accuracy | Test Accuracy | Behavior                        |
|---------------|----------------|----------------|----------------------------------|
| Underfitting   | Low             | Low             | Too simple, misses patterns       |
| Overfitting    | High            | Low             | Too complex, memorizes data       |
| Well-Fitting   | High            | High            | Balanced generalization           |

---

## 🧪 Python Code Example

We used a **non-linear dataset** (`sin(x)` + noise) and trained three models:

- `LinearRegression` → Underfitting
- `DecisionTreeRegressor(max_depth=None)` → Overfitting
- `DecisionTreeRegressor(max_depth=3)` → Well-Fitting

> See the `model_fit_demo.py` file in this repo for full code and plots.

---

## 📊 Visualization

The output graph clearly shows how each model fits the data differently, helping us visually understand the bias-variance tradeoff.

---

## 📁 Files Included

- `model_fit_demo.py` – Python script for model training and visualization
- `images/` – Graphs for each model type (underfitting, overfitting, well-fitting)
- `README.md` – Documentation for Day 1

---

## 📆 Learning Goal

This project is part of my **40 Days of ML** self-learning plan to master machine learning fundamentals with hands-on examples, visual insights, and real-world applications.

---

---

## 🧠 Next Up:  
**Bias–Variance Tradeoff** — Coming on Day 2!

---

