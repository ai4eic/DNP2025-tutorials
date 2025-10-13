# Building the Tutorial

## Jupyter Book

This tutorial is built using Jupyter Book, a modern documentation framework. There are several ways to build and view it:

### Option 1: GitHub Pages (Recommended)

The tutorial is automatically built and deployed to GitHub Pages:

- **Live Site:** [https://ai4eic.github.io/DNP2025-tutorials/](https://ai4eic.github.io/DNP2025-tutorials/)

The site is automatically updated when changes are pushed to the `main` branch.

### Option 2: Local Build

To build and view the book locally:

```bash
# Clone the repository
git clone https://github.com/ai4eic/DNP2025-tutorials.git
cd DNP2025-tutorials

# Install dependencies
pip install -r requirements.txt

# Build the book
jupyter-book build .

# Open the built book in your browser
# The HTML files are in _build/html/
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Option 3: Local Markdown Reader

You can also read the Markdown files directly:

**Using VS Code:**
1. Install the "Markdown Preview Enhanced" extension
2. Open any chapter in `chapters/` directory
3. Press `Ctrl+K V` (or `Cmd+K V` on Mac) to open preview

**Using Grip:**
```bash
# Install grip
pip install grip

# Serve the README
grip README.md 6419

# Open http://localhost:6419 in your browser
```

### Option 4: Simple Web Server

You can serve the built HTML files with any web server:

```bash
# Build the book first
jupyter-book build .

# Serve the HTML files
cd _build/html
python -m http.server 8000

# Open http://localhost:8000 in your browser
```

## Chapter Organization

The tutorial is organized as follows:

- `README.md` - Main introduction
- `SUMMARY.md` - Table of contents
- `chapters/` - Individual chapter files
  - `01-overview.md` through `21-faq.md`

## Contributing

To add or modify content:

1. Edit the relevant Markdown file in `chapters/`
2. Update `SUMMARY.md` if adding new chapters
3. Test locally using one of the methods above
4. Submit a pull request

## Viewing on GitHub

GitHub automatically renders Markdown files, so you can:

1. Navigate to the repository on GitHub
2. Browse the `chapters/` directory
3. Click on any `.md` file to view it

Start with [README.md](README.md) and follow the links in [SUMMARY.md](SUMMARY.md).

## Configuration Files

- `_config.yml` - Jupyter Book configuration (title, author, repository links, etc.)
- `_toc.yml` - Table of contents structure
- `.github/workflows/deploy-book.yml` - GitHub Actions workflow for automatic deployment

## Notes

- Math equations are rendered using MathJax
- The book supports both Markdown (`.md`) and Jupyter Notebook (`.ipynb`) files
- For production deployment, the book is automatically deployed to GitHub Pages via GitHub Actions
- The `gh-pages` branch contains the built HTML files

## Troubleshooting

**Build Errors:**
If you encounter errors during build, ensure you have the latest version of jupyter-book:
```bash
pip install --upgrade jupyter-book
```

**Missing Dependencies:**
Install all required dependencies:
```bash
pip install -r requirements.txt
```

**Math Rendering:**
LaTeX equations should render automatically with MathJax. If you have issues, check the `_config.yml` configuration.

**Clean Build:**
If you need to clean the build cache:
```bash
jupyter-book clean .
jupyter-book build .
```
