## Notebooks
If using notebooks, make sure to clear notebook output before committing changes to the repository. Alternatively, you can work in a branch off of main, and do a squash commit back into main when complete.

The `.gitattributes` file is used to assign an attribute to all notebook files, and a modification to the local `.git/config` file is made to call a function to clear out all notebook results before committing.

After cloning the repository for the first time, you can call
```bash
pipenv run notebooks_with_git
```
or, from the parent directory:
```bash
git config --local include.path ../env/.gitconfig
git add --renormalize .
```

## Branches
Use of branching on this project follows the methodology described in 
- [This branching gist](https://gist.github.com/digitaljhelms/4287848#file-gistfile1-md) by [@digitalhelms](https://gist.github.com/digitaljhelms)
- [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/) by Vincent Driessen

There are two primary branches
- `main`
- `stable`

The `main` branch is the branch under active development with new features being added and changes being made. The `stable` branch contains the latest stable version of the project code. The `main` branch is merged into `stable` once all changes are ready to be applied; typically at the end of every project sprint.

Use a branch whenever working on new features, bug fixes or changes.

### Features
Used when developing general changes, improvements or new features or models.
- Branch from `main`
- Merge into `main`
- Naming: `feature-<name>` (e.g `feature-LRmodel`)

### Bugs
Similar to features, but have a specific focus on fixing errors or issues. These changes will be deployed at the next version update.
- Branch from `main`
- Merge into `main`
- `bug-<name>` (e.g `bug-slow_MLP`)

### Hotfixes
Fixes that need to be deployed to production immediately. These are fixes to the stable branch.
- Branch from `stable`
- Merge into `stable`
- `hotfix-<name>` (e.g `fix-incorrect_directory`)

## Workflow Diagram
![Git Branching Model](https://raw.githubusercontent.com/digitaljhelms/public/master/gitflow-model.png)  
*[gitflow-model.src.key](https://github.com/digitaljhelms/public/raw/master/gitflow-model.src.key)*