# RustBlock showcase

A static, English-only landing page for RustBlock. No build step and no runtime
dependencies: Vercel serves `index.html`, `styles.css`, `script.js` and the
`assets/` folder directly.

The download cards resolve the Windows installer, portable ZIP, Linux x86_64 and
both macOS archives from the latest GitHub Release. A second request renders the
live star, fork and release-download counts in the stats strip; both responses are
cached in `sessionStorage` for ten minutes to stay well inside the unauthenticated
GitHub API limit.

The **Build** and **Run a build** buttons point at the manual
`.github/workflows/build.yml` workflow, which produces downloadable artifacts for
all four targets without creating a release.

## Preview locally

From the repository root:

```bash
cd site
python -m http.server 4173
```

Open <http://127.0.0.1:4173>.

## Deploy to Vercel

1. Import <https://github.com/Clutsy/RustBlock> in the Vercel dashboard.
2. Set **Root Directory** to `site`.
3. Framework preset: **Other**. Build command: empty. Output directory: `.`.
4. Deploy.

Or with the CLI:

```bash
npx vercel --cwd site --prod
```

`site/vercel.json` adds clean URLs, long-lived asset caching and basic security
headers. The download buttons and GitHub counters fall back gracefully to the
releases page and em dashes when the API is unavailable.
