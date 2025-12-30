# Michigrad & LLMs from Scratch

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Status](https://img.shields.io/badge/status-educational-orange.svg)

<p align="center">
  <strong>
    A collection of from-scratch implementations to master Deep Learning and Large Language Model fundamentals.
  </strong>
</p>

</div>

---

## 📖 Overview

This repository hosts educational projects designed to demystify modern Artificial Intelligence.
By building core components from the ground up, we aim to understand what happens **under the hood**
of libraries like Torch.

---

## 📂 Repository Contents

| Project | Folder | Description |
|-------|--------|-------------|
| **Michigrad** | Michigrad-from-scratch/ | A lightweight scalar-valued autograd engine implementing dynamic computational graphs and backpropagation. |
| **LLMs** | Llms-from-scratch/ | From tokenization and n-grams to self-attention and Transformer architectures. |

---

## 🚀 Installation

Python **3.8+** is recommended.

    git clone https://github.com/igna-s/Michigrad-Autograd-Engine.git
    cd Michigrad-Autograd-Engine


---

## 💻 Quick Usage Guide

### Using the Michigrad Engine

    from michigrad.engine import Value #This is the michigrad library

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b
    d = e + c
    L = d * 2.0

    L.backward()

    print(f"Loss: {L.data}")
    print(f"dL/da: {a.grad}")

---

### Exploring LLMs

    cd Llms-from-scratch
    jupyter notebook

---

## 📚 Credits & Acknowledgments

Developed as part of a **Large Language Models Workshop**.

### Special Thanks
- **Joaquín Bogado** (GitHub: @jwackito)

### Inspiration
- Andrej Karpathy’s **micrograd**

---

<div align="center">
  <p>Created with ❤️ to learn AI by breaking things.</p>
</div>
