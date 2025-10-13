# GitBook to Jupyter Book Migration

This document describes the migration from GitBook to Jupyter Book for the DNP 2025 Deep Learning Tutorials.

## Overview

The tutorial has been migrated from GitBook to Jupyter Book, a modern documentation framework built on Sphinx and MyST Markdown. This provides better integration with Jupyter notebooks, improved math rendering, and automated deployment to GitHub Pages.

## What Changed

### New Files

- **`_config.yml`**: Jupyter Book configuration (replaces `book.json`)
- **`_toc.yml`**: Table of contents structure (replaces `SUMMARY.md` format)
- **`.github/workflows/deploy-book.yml`**: GitHub Actions workflow for automatic deployment
- **`DEPLOYMENT.md`**: Documentation for deployment process
- **`MIGRATION.md`**: This file - migration notes

### Modified Files

- **`requirements.txt`**: Added `jupyter-book>=1.0.0`
- **`BUILD.md`**: Updated with Jupyter Book build instructions
- **`README.md`**: Updated to mention Jupyter Book and link to live site
- **`QUICKSTART.md`**: Updated viewing instructions
- **`.gitignore`**: Added `_build` directory

### Removed Files

- **`book.json`**: Legacy GitBook configuration (replaced by `_config.yml`)
- **`.gitbook.yaml`**: Legacy GitBook settings (replaced by `_config.yml`)

### Unchanged Files

All markdown files in `chapters/` remain unchanged and compatible with Jupyter Book:
- Chapter content is preserved exactly as-is
- Math equations (LaTeX) are still supported
- Code blocks are still supported
- All links continue to work

## Key Features

### Jupyter Book Advantages

1. **Better Math Rendering**: MathJax 3 with improved LaTeX support
2. **Notebook Support**: Native support for `.ipynb` files alongside `.md` files
3. **Modern Theme**: Clean, responsive design with dark mode
4. **Search**: Built-in full-text search
5. **GitHub Integration**: Edit buttons, issue links, repository links
6. **PDF Export**: Can generate PDF versions of the book
7. **Active Development**: Jupyter Book is actively maintained

### Deployment

- **Automatic**: Changes to `main` branch trigger automatic rebuild and deployment
- **Fast**: Builds complete in ~2-3 minutes
- **Reliable**: Uses GitHub Actions and GitHub Pages infrastructure
- **Live URL**: [https://ai4eic.github.io/DNP2025-tutorials/](https://ai4eic.github.io/DNP2025-tutorials/)

## For Contributors

### Building Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Build the book
jupyter-book build .

# View the book
open _build/html/index.html
```

### Adding Content

1. Create/edit markdown files in `chapters/`
2. Update `_toc.yml` if adding new chapters
3. Build locally to test
4. Commit and push - deployment is automatic

### Configuration

- **Book settings**: Edit `_config.yml`
- **Table of contents**: Edit `_toc.yml`
- **Theme/styling**: Modify `_config.yml` under `html` and `sphinx` sections

## Migration Notes

### Compatibility

Jupyter Book uses MyST Markdown, which is a superset of CommonMark and is fully compatible with the existing markdown:

- ✅ Standard Markdown syntax works unchanged
- ✅ LaTeX math with `$$...$$` and `$...$` works unchanged
- ✅ Code blocks with syntax highlighting work unchanged
- ✅ Links, images, tables all work unchanged
- ✅ HTML tags are supported

### TOC Structure

The `_toc.yml` structure differs from GitBook's `SUMMARY.md`:

**GitBook (SUMMARY.md):**
```markdown
* [Chapter Title](path/to/file.md)
```

**Jupyter Book (_toc.yml):**
```yaml
- file: path/to/file
  title: Chapter Title
```

The conversion preserves the chapter structure with parts/sections.

### URL Structure

The URL structure remains similar:
- `chapters/01-overview.md` → `/chapters/01-overview.html`
- Root files like `README.md` → `/README.html`

The main entry point is now `/README.html` (redirected from `/index.html`).

## Rollback Plan

If needed, you can rollback to GitBook by:

1. Reverting the migration commits
2. Restoring `book.json` and `.gitbook.yaml`
3. Removing `_config.yml` and `_toc.yml`
4. Removing the GitHub Actions workflow

However, Jupyter Book is recommended for its modern features and active maintenance.

## Support

For questions about Jupyter Book:
- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Jupyter Book GitHub](https://github.com/executablebooks/jupyter-book)
- [Community Forum](https://github.com/executablebooks/jupyter-book/discussions)

For questions about this tutorial:
- [Open an issue](https://github.com/ai4eic/DNP2025-tutorials/issues)
- See [CONTRIBUTING.md](CONTRIBUTING.md)

## Timeline

- **October 13, 2025**: Migration completed
- **Next**: Monitor deployment and gather feedback

## Credits

Migration implemented using Jupyter Book v1.0.4.post1.
