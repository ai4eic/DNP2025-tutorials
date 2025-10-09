# Building the Tutorial

## GitBook

This tutorial is designed to be viewed as a GitBook. There are several ways to build and view it:

### Option 1: GitBook.com (Recommended)

The easiest way to view this tutorial is through GitBook.com:

1. Go to [GitBook.com](https://www.gitbook.com/)
2. Import this GitHub repository
3. GitBook will automatically build and host the tutorial

### Option 2: Local Markdown Reader

Since GitBook CLI has compatibility issues with recent Node.js versions, you can read the Markdown files directly:

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

**Using mdbook (Rust-based alternative):**
```bash
# Install mdbook
cargo install mdbook

# Create mdbook configuration
# (We can add this if there's interest)
```

### Option 3: Simple Web Server

You can serve the markdown files with any web server:

```bash
# Using Python
cd /path/to/DNP2025-tutorials
python -m http.server 8000

# Open http://localhost:8000 in your browser
```

### Option 4: Jupyter Book (Alternative)

If you prefer Jupyter Book format:

```bash
# Install jupyter-book
pip install jupyter-book

# Build the book (requires conversion to Jupyter Book structure)
# jupyter-book build .
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

## Notes

- The `book.json` file is configured for GitBook but may need updates for compatibility with newer GitBook versions
- For production deployment, we recommend using GitBook.com or GitHub Pages with Jekyll/Hugo
- MathJax/KaTeX is configured in `book.json` for rendering equations

## Troubleshooting

**GitBook CLI Issues:**
The legacy GitBook CLI has known compatibility issues with Node.js v12+. Use GitBook.com or alternative readers instead.

**Missing Plugins:**
Some GitBook plugins in `book.json` may not be available. You can remove them from the configuration if needed.

**Math Rendering:**
LaTeX equations should render in GitBook.com and most Markdown previews. For local viewing, ensure MathJax or KaTeX support is enabled.
