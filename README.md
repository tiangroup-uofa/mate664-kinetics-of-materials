# CHE 318 — Mass Transfer

This repository contains **open-source course materials** for **CHE 318 (Mass Transfer)** at University of Alberta

The course website, lecture notes, and slides are built using
[**Quarto**](https://quarto.org) and interactive computing tools such as
[**Marimo**](https://docs.marimo.io).

If you spot a typo, unclear explanation, or small issue, **feel free to open an issue or submit a pull request** — contributions are welcome.

---

## Build the site locally

To build and preview the course materials on your own machine:

### 1. Install Quarto
Follow the official installation instructions:  
👉 https://quarto.org/docs/get-started/

### 2. Install `uv`
`uv` is a fast Python package manager used in this project.

👉 https://docs.astral.sh/uv/

### 3. Install Marimo
From the repository root:

```bash
uv pip install marimo
```

### 4. Render the output

```bash
quarto render .
```

### 5. Preview the contents

```bash
quarto preview .
```
A browser page will be opened for the rendered websites.
If you are only modifying the `.qmd` source codes, the website can track these changes on the fly. 
Otherwise if the marimo notebooks are changing, please rerun `quarto render .`.
