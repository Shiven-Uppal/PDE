# GitHub upload notes

This folder is ready to initialize as the root of a repository named `pde-pm25-adr`. The GitHub connection available during packaging could read the account but did not expose a create-repository operation or binary file upload. Create an empty repository in the GitHub UI, then from this folder run:

```bash
git init
git add .
git commit -m "Add Delhi PM2.5 ADR project and supporting materials"
git branch -M main
git remote add origin https://github.com/Shiven-Uppal/pde-pm25-adr.git
git push -u origin main
```

Large raw CPCB and OSM files are intentionally git-ignored. Keep them in the local archive or use Git LFS / a data repository for distribution. The processed data tables and scripts remain in the repository.
