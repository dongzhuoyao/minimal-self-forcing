# Using Markdown Files as Central Guidance in Cursor

This guide explains how to make markdown files serve as central guidance for Cursor AI assistance.

## Method 1: `.cursorrules` File (Recommended)

The `.cursorrules` file is automatically read by Cursor and provides context for AI assistance.

### How It Works

1. **Automatic Loading**: Cursor automatically reads `.cursorrules` from your project root
2. **Always Available**: The content is always available as context for AI assistance
3. **No Manual Reference Needed**: You don't need to @-mention it

### Current Setup

We've created `.cursorrules` in the project root that includes:
- Project overview and key concepts
- References to important documentation files
- Code style guidelines
- Common tasks and commands
- AI assistant guidelines

### Updating `.cursorrules`

Simply edit `.cursorrules` to add or modify guidance. The file uses markdown format and can include:
- Project structure
- Coding conventions
- Key concepts and algorithms
- Common workflows
- Important notes and warnings

## Method 2: @-Mentioning Files

You can reference specific markdown files in your conversations:

```
@markdown/HOW_TO_RUN.md How do I train the model?
```

This makes Cursor read that specific file for context.

## Method 3: Prominent Documentation Files

Cursor often automatically considers:
- `README.md` in the project root
- Files in commonly named directories like `docs/`, `markdown/`, `documentation/`

## Best Practices

### 1. Keep `.cursorrules` Concise
- Include key information and references
- Link to detailed documentation rather than duplicating it
- Focus on what the AI needs to know to help effectively

### 2. Organize Documentation
- Use a `markdown/` or `docs/` folder for detailed guides
- Keep `README.md` as a quick reference
- Create focused guides for specific topics

### 3. Update Regularly
- Keep `.cursorrules` in sync with project changes
- Update documentation when adding new features
- Remove outdated information

### 4. Use Clear Structure
- Use headers and sections for easy navigation
- Include code examples
- Reference specific files and line numbers when helpful

## Current Project Structure

```
├── .cursorrules          # Central guidance (automatically loaded)
├── README.md             # Project overview
└── markdown/             # Detailed documentation
    ├── HOW_TO_RUN.md     # Complete operations guide
    ├── QUICK_START.md    # Quick start examples
    ├── ALGORITHM_SUMMARY.md  # Algorithm details
    ├── GENERATION_GUIDE.md   # Generation guide
    └── ...               # Other guides
```

## Example Usage

### In Cursor Chat

You can ask questions and Cursor will automatically use `.cursorrules`:

```
How do I train the model?
```

Cursor will reference `.cursorrules` and the documentation it mentions.

### Explicit File Reference

For specific topics, you can reference files directly:

```
@markdown/HOW_TO_RUN.md What are all the training arguments?
```

## Tips

1. **Start with `.cursorrules`**: This is the most effective way to provide central guidance
2. **Reference, Don't Duplicate**: Link to detailed docs rather than copying everything
3. **Keep It Updated**: Update `.cursorrules` when project structure or conventions change
4. **Use Clear Sections**: Organize information with headers for easy scanning
5. **Include Examples**: Code examples help the AI understand context better

## Verification

To verify `.cursorrules` is working:

1. Open Cursor chat
2. Ask a question about the project
3. The AI should reference information from `.cursorrules`
4. It should also know about your documentation structure

## Troubleshooting

**Cursor not using `.cursorrules`?**
- Ensure the file is named exactly `.cursorrules` (with the dot)
- Place it in the project root directory
- Restart Cursor if needed

**Want to reference specific files?**
- Use @-mention syntax: `@filename.md`
- Or use relative paths: `@markdown/HOW_TO_RUN.md`

**Need to update guidance?**
- Edit `.cursorrules` directly
- Changes take effect immediately (may need to restart Cursor)
