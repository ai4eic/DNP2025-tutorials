# Deployment Guide

This document explains how the Jupyter Book is built and deployed to GitHub Pages.

## Automatic Deployment

The tutorial is automatically built and deployed using GitHub Actions whenever changes are pushed to the `main` branch.

### Deployment Workflow

1. **Trigger**: Push to `main` branch or manual workflow dispatch
2. **Build**: GitHub Actions builds the Jupyter Book using `jupyter-book build .`
3. **Deploy**: Built HTML files are deployed to the `gh-pages` branch
4. **Publish**: GitHub Pages serves the content from `gh-pages` branch

### Workflow File

The deployment is configured in `.github/workflows/deploy-book.yml`:

```yaml
name: Build and Deploy Jupyter Book

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pages: write
      id-token: write
```

## GitHub Pages Configuration

### Repository Settings

To enable GitHub Pages for this repository:

1. Go to repository **Settings** → **Pages**
2. Under **Source**, select:
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
3. Click **Save**

The site will be available at: `https://ai4eic.github.io/DNP2025-tutorials/`

### Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to the `gh-pages` branch with your domain
2. Configure your DNS provider to point to GitHub Pages
3. Update the `html.baseurl` in `_config.yml`

## Manual Deployment

If you need to deploy manually:

```bash
# Build the book
jupyter-book build .

# Install ghp-import if not already installed
pip install ghp-import

# Push to gh-pages branch
ghp-import -n -p -f _build/html
```

## Deployment Status

You can check the deployment status in the repository:

- **Actions tab**: View workflow runs and build logs
- **Deployments**: See active deployments and history

## Troubleshooting

### Build Fails

Check the GitHub Actions logs for errors:
1. Go to **Actions** tab
2. Click on the failed workflow run
3. Review the build logs

Common issues:
- **Missing dependencies**: Update `requirements.txt` in the workflow
- **Invalid markdown**: Check for syntax errors in `.md` files
- **Missing files**: Ensure all files referenced in `_toc.yml` exist

### Pages Not Updating

If changes aren't appearing:
1. Check workflow succeeded in **Actions** tab
2. Verify `gh-pages` branch has new commits
3. Wait 1-2 minutes for GitHub Pages to update
4. Clear browser cache

### 404 Errors

If you get 404 errors:
1. Ensure `gh-pages` branch exists
2. Check GitHub Pages settings
3. Verify the `.nojekyll` file is present in `gh-pages` branch

## Build Cache

The workflow uses caching to speed up builds:
- **pip cache**: Python packages
- **jupyter-cache**: Notebook execution results

To clear the cache, manually delete it in the Actions cache settings.

## Security

The workflow uses GitHub's built-in `GITHUB_TOKEN` which has:
- **Read** access to repository
- **Write** access to `gh-pages` branch
- **Write** access to GitHub Pages

No additional secrets are required.

## Monitoring

Monitor deployment health:
- **GitHub Actions**: Check workflow status
- **GitHub Pages**: Monitor uptime and availability
- **Analytics**: Add Google Analytics or similar (optional)

## Local Testing

Before pushing changes, test locally:

```bash
# Clean previous build
jupyter-book clean .

# Build the book
jupyter-book build .

# View in browser
open _build/html/index.html
```

This ensures your changes will build successfully in CI/CD.

## Next Steps

- Review [BUILD.md](BUILD.md) for local build instructions
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Check [_config.yml](_config.yml) for Jupyter Book configuration
