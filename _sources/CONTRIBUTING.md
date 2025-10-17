# Contributing to DNP 2025 Deep Learning Tutorials

Thank you for your interest in contributing to the DNP 2025 Deep Learning Tutorials! This document provides guidelines for contributing to this project.

## Ways to Contribute

### 1. Report Issues

If you find errors, typos, or unclear explanations:
- Check if the issue already exists in [GitHub Issues](https://github.com/ai4eic/DNP2025-tutorials/issues)
- If not, create a new issue with:
  - Clear description of the problem
  - Chapter/section reference
  - Suggested fix (if applicable)

### 2. Improve Documentation

Help improve the tutorial content:
- Fix typos and grammatical errors
- Clarify confusing explanations
- Add examples or illustrations
- Improve code comments
- Update outdated information

### 3. Add Content

Contribute new material:
- Complete placeholder chapters (05, 06, 07, 09-11, 13-18, 21)
- Add Jupyter notebooks with examples
- Create supplementary exercises
- Add visualization examples
- Contribute sample datasets

### 4. Code Contributions

Enhance the codebase:
- Add working code examples
- Improve existing implementations
- Add unit tests
- Optimize performance
- Fix bugs

### 5. Review Pull Requests

Help review contributions:
- Test code examples
- Verify technical accuracy
- Check for clarity and completeness
- Suggest improvements

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DNP2025-tutorials.git
   cd DNP2025-tutorials
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ai4eic/DNP2025-tutorials.git
   ```

### Set Up Environment

Follow the instructions in [chapters/02-setup.md](chapters/02-setup.md) to set up your development environment.

### Create a Branch

Create a feature branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `fix/typo-in-chapter-08`
- `feature/add-vae-examples`
- `docs/improve-setup-instructions`

## Making Changes

### Writing Guidelines

**Markdown Style:**
- Use clear, concise language
- Break long paragraphs into shorter ones
- Use headers hierarchically (h1 → h2 → h3)
- Include code blocks with language specification
- Add alt text for images

**Code Style:**
- Follow PEP 8 for Python code
- Include docstrings for functions
- Add comments for complex logic
- Use descriptive variable names
- Keep functions focused and small

**Technical Content:**
- Ensure physics accuracy
- Verify code examples run correctly
- Include necessary imports
- Provide expected outputs
- Cite sources appropriately

### Testing Your Changes

**For Documentation:**
1. Preview Markdown locally (see [BUILD.md](BUILD.md))
2. Check all links work
3. Verify code examples run
4. Test on different Markdown readers

**For Code:**
1. Run code examples end-to-end
2. Test with different Python versions (3.8+)
3. Verify dependencies in requirements.txt
4. Check PyTorch and TensorFlow compatibility

### Commit Guidelines

Write clear commit messages:

**Format:**
```
<type>: <short summary>

<detailed description>

<issue reference>
```

**Types:**
- `feat:` New feature or content
- `fix:` Bug fix or correction
- `docs:` Documentation changes
- `style:` Formatting changes
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**
```
feat: Add VAE implementation for FCAL shower generation

Implement Variational Autoencoder architecture with:
- Encoder network for latent space encoding
- Decoder network for shower reconstruction
- Training loop with KL divergence loss
- Visualization of latent space

Closes #42
```

```
fix: Correct energy normalization in preprocessing

The previous normalization used incorrect scale factor.
Updated to match FCAL energy range (0.1-12 GeV).

Fixes #56
```

## Submitting Changes

### Pull Request Process

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request:**
   - Go to GitHub and create a PR from your fork
   - Fill out the PR template
   - Link related issues
   - Request review from maintainers

### PR Checklist

Before submitting, ensure:
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts
- [ ] Changes are focused and minimal

### Review Process

**What to Expect:**
1. Maintainers will review your PR
2. May request changes or clarifications
3. Address feedback and update PR
4. Once approved, maintainers will merge

**Timeline:**
- Initial review: within 1 week
- Follow-up reviews: within 3 days
- Merge: after approval and CI passes

## Content Guidelines

### Chapter Structure

Each chapter should include:
1. **Introduction**: Overview and learning objectives
2. **Theory**: Conceptual background
3. **Examples**: Code demonstrations
4. **Exercises**: Practice problems
5. **Summary**: Key takeaways
6. **Next Steps**: Links to related chapters

### Code Examples

**Requirements:**
- Executable and tested
- Include necessary imports
- Show expected output
- Handle errors gracefully
- Include comments

**Example Template:**
```python
"""
Brief description of what this code does.
"""
import numpy as np
import torch

def example_function(data):
    """
    Brief function description.
    
    Args:
        data: Input description
        
    Returns:
        Output description
    """
    # Implementation with comments
    result = process(data)
    return result

# Usage example
if __name__ == "__main__":
    # Example data
    data = np.random.rand(100, 32, 32)
    
    # Process
    output = example_function(data)
    print(f"Output shape: {output.shape}")
```

### Physics Content

**Ensure:**
- Physical accuracy
- Proper terminology
- Unit consistency
- Citation of sources
- Clear definitions

**Avoid:**
- Unexplained jargon
- Unsupported claims
- Mixing conventions
- Missing uncertainties

## Style Guide

### Language

- Use inclusive language
- Write in present tense
- Prefer active voice
- Be concise and clear
- Define acronyms on first use

### Formatting

**Headers:**
- Use sentence case
- No period at end
- Hierarchical structure

**Lists:**
- Parallel structure
- Consistent punctuation
- Ordered for sequences
- Unordered for groups

**Code Blocks:**
- Specify language
- Include context
- Show input/output
- Keep readable width

**Equations:**
Use LaTeX for math:
```latex
$$E = mc^2$$
```

### References

Cite sources using:
- Author names and year
- arXiv IDs for preprints
- DOIs for published papers
- URLs for web resources

Example:
```markdown
Deep learning has shown promise in HEP [1,2].

[1] Guest et al., "Deep Learning and its Application to LHC Physics," arXiv:1806.11484 (2018)
[2] Albertsson et al., "Machine Learning in HEP Community White Paper," arXiv:1807.02876 (2018)
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Give constructive feedback
- Credit others' work
- Maintain professionalism

### Communication

**GitHub Issues:**
- Search before creating
- Use clear titles
- Provide context
- Stay on topic

**Pull Requests:**
- Respond to reviews
- Be open to feedback
- Explain your changes
- Ask questions

**Discussions:**
- Use appropriate forums
- Be helpful and patient
- Share knowledge
- Acknowledge contributions

## Recognition

Contributors will be:
- Listed in repository contributors
- Acknowledged in tutorial credits
- Invited to collaboration meetings (if regular contributors)
- Co-authors on related publications (for substantial contributions)

## Getting Help

**Questions?**
- Check [FAQ](chapters/21-faq.md)
- Search [Issues](https://github.com/ai4eic/DNP2025-tutorials/issues)
- Ask in GitHub Discussions
- Contact maintainers

**Resources:**
- [README.md](README.md) - Project overview
- [BUILD.md](BUILD.md) - Building instructions
- [chapters/02-setup.md](chapters/02-setup.md) - Environment setup
- [chapters/19-references.md](chapters/19-references.md) - Additional resources

## License

By contributing, you agree that your contributions will be licensed under the same [CC0 1.0 Universal](LICENSE) license that covers the project.

---

Thank you for contributing to the DNP 2025 Deep Learning Tutorials! Your contributions help make deep learning more accessible to the nuclear physics community.

**Questions?** Open an issue or contact the maintainers.
